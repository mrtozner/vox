//! Helpers shared by the WebSocket endpoints (`ws_listen`, `ws_speak`,
//! `ws_converse`, `live_talk`).

/// Interpret raw bytes as f32 little-endian PCM samples.
pub(super) fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub(super) fn encode_wav_chunk(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    // 16-bit PCM: Chrome's HTMLAudioElement cannot decode 32-bit float WAV.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer =
            hound::WavWriter::new(&mut buf, spec).map_err(|e| format!("WAV init: {e}"))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample_i16 = (clamped * i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| format!("WAV write: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("WAV finalize: {e}"))?;
    }
    Ok(buf.into_inner())
}

// ─── Server-wide hooks for session loops that do not know `ServerState` ───
//
// `live_talk` and `ws_converse` run their sessions over a generic transport
// and plain backends, so they can be driven without a server. The pieces
// below attach the server-wide behaviour at the socket handler: the 1001
// close frame on shutdown and the shared inference limit.

/// Messages buffered between a session and the socket writer.
const SHUTDOWN_RELAY_BUFFER: usize = 64;

/// Wrap a WebSocket's two halves so the session ends cleanly on shutdown.
///
/// Returns a sink and a stream to hand to the session loop, and the relay
/// task that owns the real sink. When `shutdown` is cancelled the stream
/// ends (the session sees the client as gone and winds down) and the relay
/// sends [`super::shutdown_close_message`]. Await the relay after the
/// session returns so the last frames are written before the handler exits.
pub(super) fn with_shutdown<S, R>(
    mut sink: S,
    stream: R,
    shutdown: tokio_util::sync::CancellationToken,
) -> (
    tokio_util::sync::PollSender<axum::extract::ws::Message>,
    impl futures_util::Stream<Item = R::Item> + Unpin,
    tokio::task::JoinHandle<()>,
)
where
    S: futures_util::Sink<axum::extract::ws::Message> + Send + Unpin + 'static,
    R: futures_util::Stream + Unpin,
{
    use futures_util::{SinkExt, StreamExt};

    let (tx, mut rx) = tokio::sync::mpsc::channel(SHUTDOWN_RELAY_BUFFER);
    let relay = {
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    () = shutdown.cancelled() => {
                        let _ = sink.send(super::shutdown_close_message()).await;
                        break;
                    }
                    message = rx.recv() => match message {
                        Some(message) => {
                            if sink.send(message).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    },
                }
            }
            let _ = sink.close().await;
        })
    };
    let stream = stream.take_until(Box::pin(shutdown.cancelled_owned()));
    (tokio_util::sync::PollSender::new(tx), stream, relay)
}

/// An STT backend that takes a server inference slot for each call.
pub(super) struct LimitedStt {
    pub inner: std::sync::Arc<dyn crate::traits::SttBackend>,
    pub state: std::sync::Arc<super::ServerState>,
}

#[async_trait::async_trait]
impl crate::traits::SttBackend for LimitedStt {
    async fn transcribe(
        &self,
        audio: &crate::Utterance,
    ) -> Result<crate::SttResult, crate::VoxError> {
        let Some(_permit) = self.state.inference_permit_wait().await else {
            return Err(crate::VoxError::Stt("server is shutting down".into()));
        };
        self.inner.transcribe(audio).await
    }
}

/// A TTS backend that takes a server inference slot for each call.
pub(super) struct LimitedTts {
    pub inner: std::sync::Arc<dyn crate::traits::TtsBackend>,
    pub state: std::sync::Arc<super::ServerState>,
}

#[async_trait::async_trait]
impl crate::traits::TtsBackend for LimitedTts {
    async fn synthesize(
        &self,
        request: &crate::TtsRequest,
    ) -> Result<crate::TtsOutput, crate::VoxError> {
        let Some(_permit) = self.state.inference_permit_wait().await else {
            return Err(crate::VoxError::Tts("server is shutting down".into()));
        };
        self.inner.synthesize(request).await
    }

    fn list_voices(&self) -> Vec<crate::types::VoiceInfo> {
        self.inner.list_voices()
    }

    fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }
}

#[cfg(test)]
mod shutdown_relay_tests {
    use axum::extract::ws::Message;
    use futures_util::{SinkExt, StreamExt};
    use tokio_util::sync::{CancellationToken, PollSender};

    use super::with_shutdown;

    type Incoming = Result<Message, std::convert::Infallible>;

    #[tokio::test]
    async fn relay_forwards_then_sends_1001_on_shutdown() {
        let (socket_tx, mut socket_rx) = tokio::sync::mpsc::channel::<Message>(8);
        let incoming = futures_util::stream::pending::<Incoming>();
        let token = CancellationToken::new();

        let (mut sink, mut stream, relay) =
            with_shutdown(PollSender::new(socket_tx), incoming, token.clone());

        sink.send(Message::Text("hello".into())).await.unwrap();
        match socket_rx.recv().await {
            Some(Message::Text(text)) => assert_eq!(text.as_str(), "hello"),
            other => panic!("expected the forwarded text, got {other:?}"),
        }

        token.cancel();
        assert!(
            stream.next().await.is_none(),
            "the session's stream ends on shutdown"
        );
        match socket_rx.recv().await {
            Some(Message::Close(Some(frame))) => {
                assert_eq!(frame.code, axum::extract::ws::close_code::AWAY);
            }
            other => panic!("expected a 1001 close frame, got {other:?}"),
        }
        relay.await.unwrap();
    }

    #[tokio::test]
    async fn relay_ends_when_the_session_drops_its_sink() {
        let (socket_tx, mut socket_rx) = tokio::sync::mpsc::channel::<Message>(8);
        let incoming = futures_util::stream::pending::<Incoming>();
        let (sink, _stream, relay) = with_shutdown(
            PollSender::new(socket_tx),
            incoming,
            CancellationToken::new(),
        );

        drop(sink);
        relay.await.unwrap();
        assert!(
            socket_rx.recv().await.is_none(),
            "no close frame was invented"
        );
    }
}
