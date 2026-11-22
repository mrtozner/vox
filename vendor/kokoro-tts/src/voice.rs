use crate::KokoroError;

//noinspection SpellCheckingInspection
#[derive(Copy, Clone, Debug)]
pub enum Voice {
    // v1.0
    ZmYunyang(f32),
    ZfXiaoni(f32),
    AfJessica(f32),
    BfLily(f32),
    ZfXiaobei(f32),
    ZmYunxia(f32),
    AfHeart(f32),
    BfEmma(f32),
    AmPuck(f32),
    BfAlice(f32),
    HfAlpha(f32),
    BfIsabella(f32),
    AfNova(f32),
    AmFenrir(f32),
    EmAlex(f32),
    ImNicola(f32),
    PmAlex(f32),
    AfAlloy(f32),
    ZmYunxi(f32),
    AfSarah(f32),
    JfNezumi(f32),
    BmDaniel(f32),
    JfTebukuro(f32),
    JfAlpha(f32),
    JmKumo(f32),
    EmSanta(f32),
    AmLiam(f32),
    AmSanta(f32),
    AmEric(f32),
    BmFable(f32),
    AfBella(f32),
    BmLewis(f32),
    PfDora(f32),
    AfNicole(f32),
    BmGeorge(f32),
    AmOnyx(f32),
    HmPsi(f32),
    HfBeta(f32),
    HmOmega(f32),
    ZfXiaoxiao(f32),
    FfSiwis(f32),
    EfDora(f32),
    AfAoede(f32),
    AmEcho(f32),
    AmMichael(f32),
    AfKore(f32),
    ZfXiaoyi(f32),
    JfGongitsune(f32),
    AmAdam(f32),
    IfSara(f32),
    AfSky(f32),
    PmSanta(f32),
    AfRiver(f32),
    ZmYunjian(f32),

    // v1.1
    Zm029(i32),
    Zf048(i32),
    Zf008(i32),
    Zm014(i32),
    Zf003(i32),
    Zf047(i32),
    Zm080(i32),
    Zf094(i32),
    Zf046(i32),
    Zm054(i32),
    Zf001(i32),
    Zm062(i32),
    BfVale(i32),
    Zf044(i32),
    Zf005(i32),
    Zf028(i32),
    Zf059(i32),
    Zm030(i32),
    Zf074(i32),
    Zm009(i32),
    Zf004(i32),
    Zf021(i32),
    Zm095(i32),
    Zm041(i32),
    Zf087(i32),
    Zf039(i32),
    Zm031(i32),
    Zf007(i32),
    Zf038(i32),
    Zf092(i32),
    Zm056(i32),
    Zf099(i32),
    Zm010(i32),
    Zm069(i32),
    Zm016(i32),
    Zm068(i32),
    Zf083(i32),
    Zf093(i32),
    Zf006(i32),
    Zf026(i32),
    Zm053(i32),
    Zm064(i32),
    AfSol(i32),
    Zf042(i32),
    Zf084(i32),
    Zf073(i32),
    Zf067(i32),
    Zm025(i32),
    Zm020(i32),
    Zm050(i32),
    Zf070(i32),
    Zf002(i32),
    Zf032(i32),
    Zm091(i32),
    Zm066(i32),
    Zm089(i32),
    Zm034(i32),
    Zm100(i32),
    Zf086(i32),
    Zf040(i32),
    Zm011(i32),
    Zm098(i32),
    Zm015(i32),
    Zf051(i32),
    Zm065(i32),
    Zf076(i32),
    Zf036(i32),
    Zm033(i32),
    Zf018(i32),
    Zf017(i32),
    Zf049(i32),
    AfMaple(i32),
    Zm082(i32),
    Zm057(i32),
    Zf079(i32),
    Zf022(i32),
    Zm063(i32),
    Zf060(i32),
    Zf019(i32),
    Zm097(i32),
    Zm096(i32),
    Zf023(i32),
    Zf027(i32),
    Zf085(i32),
    Zf077(i32),
    Zm035(i32),
    Zf088(i32),
    Zf024(i32),
    Zf072(i32),
    Zm055(i32),
    Zm052(i32),
    Zf071(i32),
    Zm061(i32),
    Zf078(i32),
    Zm013(i32),
    Zm081(i32),
    Zm037(i32),
    Zf090(i32),
    Zf043(i32),
    Zm058(i32),
    Zm012(i32),
    Zm045(i32),
    Zf075(i32),
}

impl Voice {
    //noinspection SpellCheckingInspection
    pub(super) fn get_name(&self) -> &str {
        match self {
            Self::ZmYunyang(_) => "zm_yunyang",
            Self::ZfXiaoni(_) => "zf_xiaoni",
            Self::AfJessica(_) => "af_jessica",
            Self::BfLily(_) => "bf_lily",
            Self::ZfXiaobei(_) => "zf_xiaobei",
            Self::ZmYunxia(_) => "zm_yunxia",
            Self::AfHeart(_) => "af_heart",
            Self::BfEmma(_) => "bf_emma",
            Self::AmPuck(_) => "am_puck",
            Self::BfAlice(_) => "bf_alice",
            Self::HfAlpha(_) => "hf_alpha",
            Self::BfIsabella(_) => "bf_isabella",
            Self::AfNova(_) => "af_nova",
            Self::AmFenrir(_) => "am_fenrir",
            Self::EmAlex(_) => "em_alex",
            Self::ImNicola(_) => "im_nicola",
            Self::PmAlex(_) => "pm_alex",
            Self::AfAlloy(_) => "af_alloy",
            Self::ZmYunxi(_) => "zm_yunxi",
            Self::AfSarah(_) => "af_sarah",
            Self::JfNezumi(_) => "jf_nezumi",
            Self::BmDaniel(_) => "bm_daniel",
            Self::JfTebukuro(_) => "jf_tebukuro",
            Self::JfAlpha(_) => "jf_alpha",
            Self::JmKumo(_) => "jm_kumo",
            Self::EmSanta(_) => "em_santa",
            Self::AmLiam(_) => "am_liam",
            Self::AmSanta(_) => "am_santa",
            Self::AmEric(_) => "am_eric",
            Self::BmFable(_) => "bm_fable",
            Self::AfBella(_) => "af_bella",
            Self::BmLewis(_) => "bm_lewis",
            Self::PfDora(_) => "pf_dora",
            Self::AfNicole(_) => "af_nicole",
            Self::BmGeorge(_) => "bm_george",
            Self::AmOnyx(_) => "am_onyx",
            Self::HmPsi(_) => "hm_psi",
            Self::HfBeta(_) => "hf_beta",
            Self::HmOmega(_) => "hm_omega",
            Self::ZfXiaoxiao(_) => "zf_xiaoxiao",
            Self::FfSiwis(_) => "ff_siwis",
            Self::EfDora(_) => "ef_dora",
            Self::AfAoede(_) => "af_aoede",
            Self::AmEcho(_) => "am_echo",
            Self::AmMichael(_) => "am_michael",
            Self::AfKore(_) => "af_kore",
            Self::ZfXiaoyi(_) => "zf_xiaoyi",
            Self::JfGongitsune(_) => "jf_gongitsune",
            Self::AmAdam(_) => "am_adam",
            Self::IfSara(_) => "if_sara",
            Self::AfSky(_) => "af_sky",
            Self::PmSanta(_) => "pm_santa",
            Self::AfRiver(_) => "af_river",
            Self::ZmYunjian(_) => "zm_yunjian",
            Self::Zm029(_) => "zm_029",
            Self::Zf048(_) => "zf_048",
            Self::Zf008(_) => "zf_008",
            Self::Zm014(_) => "zm_014",
            Self::Zf003(_) => "zf_003",
            Self::Zf047(_) => "zf_047",
            Self::Zm080(_) => "zm_080",
            Self::Zf094(_) => "zf_094",
            Self::Zf046(_) => "zf_046",
            Self::Zm054(_) => "zm_054",
            Self::Zf001(_) => "zf_001",
            Self::Zm062(_) => "zm_062",
            Self::BfVale(_) => "bf_vale",
            Self::Zf044(_) => "zf_044",
            Self::Zf005(_) => "zf_005",
            Self::Zf028(_) => "zf_028",
            Self::Zf059(_) => "zf_059",
            Self::Zm030(_) => "zm_030",
            Self::Zf074(_) => "zf_074",
            Self::Zm009(_) => "zm_009",
            Self::Zf004(_) => "zf_004",
            Self::Zf021(_) => "zf_021",
            Self::Zm095(_) => "zm_095",
            Self::Zm041(_) => "zm_041",
            Self::Zf087(_) => "zf_087",
            Self::Zf039(_) => "zf_039",
            Self::Zm031(_) => "zm_031",
            Self::Zf007(_) => "zf_007",
            Self::Zf038(_) => "zf_038",
            Self::Zf092(_) => "zf_092",
            Self::Zm056(_) => "zm_056",
            Self::Zf099(_) => "zf_099",
            Self::Zm010(_) => "zm_010",
            Self::Zm069(_) => "zm_069",
            Self::Zm016(_) => "zm_016",
            Self::Zm068(_) => "zm_068",
            Self::Zf083(_) => "zf_083",
            Self::Zf093(_) => "zf_093",
            Self::Zf006(_) => "zf_006",
            Self::Zf026(_) => "zf_026",
            Self::Zm053(_) => "zm_053",
            Self::Zm064(_) => "zm_064",
            Self::AfSol(_) => "af_sol",
            Self::Zf042(_) => "zf_042",
            Self::Zf084(_) => "zf_084",
            Self::Zf073(_) => "zf_073",
            Self::Zf067(_) => "zf_067",
            Self::Zm025(_) => "zm_025",
            Self::Zm020(_) => "zm_020",
            Self::Zm050(_) => "zm_050",
            Self::Zf070(_) => "zf_070",
            Self::Zf002(_) => "zf_002",
            Self::Zf032(_) => "zf_032",
            Self::Zm091(_) => "zm_091",
            Self::Zm066(_) => "zm_066",
            Self::Zm089(_) => "zm_089",
            Self::Zm034(_) => "zm_034",
            Self::Zm100(_) => "zm_100",
            Self::Zf086(_) => "zf_086",
            Self::Zf040(_) => "zf_040",
            Self::Zm011(_) => "zm_011",
            Self::Zm098(_) => "zm_098",
            Self::Zm015(_) => "zm_015",
            Self::Zf051(_) => "zf_051",
            Self::Zm065(_) => "zm_065",
            Self::Zf076(_) => "zf_076",
            Self::Zf036(_) => "zf_036",
            Self::Zm033(_) => "zm_033",
            Self::Zf018(_) => "zf_018",
            Self::Zf017(_) => "zf_017",
            Self::Zf049(_) => "zf_049",
            Self::AfMaple(_) => "af_maple",
            Self::Zm082(_) => "zm_082",
            Self::Zm057(_) => "zm_057",
            Self::Zf079(_) => "zf_079",
            Self::Zf022(_) => "zf_022",
            Self::Zm063(_) => "zm_063",
            Self::Zf060(_) => "zf_060",
            Self::Zf019(_) => "zf_019",
            Self::Zm097(_) => "zm_097",
            Self::Zm096(_) => "zm_096",
            Self::Zf023(_) => "zf_023",
            Self::Zf027(_) => "zf_027",
            Self::Zf085(_) => "zf_085",
            Self::Zf077(_) => "zf_077",
            Self::Zm035(_) => "zm_035",
            Self::Zf088(_) => "zf_088",
            Self::Zf024(_) => "zf_024",
            Self::Zf072(_) => "zf_072",
            Self::Zm055(_) => "zm_055",
            Self::Zm052(_) => "zm_052",
            Self::Zf071(_) => "zf_071",
            Self::Zm061(_) => "zm_061",
            Self::Zf078(_) => "zf_078",
            Self::Zm013(_) => "zm_013",
            Self::Zm081(_) => "zm_081",
            Self::Zm037(_) => "zm_037",
            Self::Zf090(_) => "zf_090",
            Self::Zf043(_) => "zf_043",
            Self::Zm058(_) => "zm_058",
            Self::Zm012(_) => "zm_012",
            Self::Zm045(_) => "zm_045",
            Self::Zf075(_) => "zf_075",
        }
    }

    pub(super) fn is_v10_supported(&self) -> bool {
        matches!(
            self,
            Self::ZmYunyang(_)
                | Self::ZfXiaoni(_)
                | Self::AfJessica(_)
                | Self::BfLily(_)
                | Self::ZfXiaobei(_)
                | Self::ZmYunxia(_)
                | Self::AfHeart(_)
                | Self::BfEmma(_)
                | Self::AmPuck(_)
                | Self::BfAlice(_)
                | Self::HfAlpha(_)
                | Self::BfIsabella(_)
                | Self::AfNova(_)
                | Self::AmFenrir(_)
                | Self::EmAlex(_)
                | Self::ImNicola(_)
                | Self::PmAlex(_)
                | Self::AfAlloy(_)
                | Self::ZmYunxi(_)
                | Self::AfSarah(_)
                | Self::JfNezumi(_)
                | Self::BmDaniel(_)
                | Self::JfTebukuro(_)
                | Self::JfAlpha(_)
                | Self::JmKumo(_)
                | Self::EmSanta(_)
                | Self::AmLiam(_)
                | Self::AmSanta(_)
                | Self::AmEric(_)
                | Self::BmFable(_)
                | Self::AfBella(_)
                | Self::BmLewis(_)
                | Self::PfDora(_)
                | Self::AfNicole(_)
                | Self::BmGeorge(_)
                | Self::AmOnyx(_)
                | Self::HmPsi(_)
                | Self::HfBeta(_)
                | Self::HmOmega(_)
                | Self::ZfXiaoxiao(_)
                | Self::FfSiwis(_)
                | Self::EfDora(_)
                | Self::AfAoede(_)
                | Self::AmEcho(_)
                | Self::AmMichael(_)
                | Self::AfKore(_)
                | Self::ZfXiaoyi(_)
                | Self::JfGongitsune(_)
                | Self::AmAdam(_)
                | Self::IfSara(_)
                | Self::AfSky(_)
                | Self::PmSanta(_)
                | Self::AfRiver(_)
                | Self::ZmYunjian(_)
        )
    }

    pub(super) fn is_v11_supported(&self) -> bool {
        matches!(
            self,
            Self::Zm029(_)
                | Self::Zf048(_)
                | Self::Zf008(_)
                | Self::Zm014(_)
                | Self::Zf003(_)
                | Self::Zf047(_)
                | Self::Zm080(_)
                | Self::Zf094(_)
                | Self::Zf046(_)
                | Self::Zm054(_)
                | Self::Zf001(_)
                | Self::Zm062(_)
                | Self::BfVale(_)
                | Self::Zf044(_)
                | Self::Zf005(_)
                | Self::Zf028(_)
                | Self::Zf059(_)
                | Self::Zm030(_)
                | Self::Zf074(_)
                | Self::Zm009(_)
                | Self::Zf004(_)
                | Self::Zf021(_)
                | Self::Zm095(_)
                | Self::Zm041(_)
                | Self::Zf087(_)
                | Self::Zf039(_)
                | Self::Zm031(_)
                | Self::Zf007(_)
                | Self::Zf038(_)
                | Self::Zf092(_)
                | Self::Zm056(_)
                | Self::Zf099(_)
                | Self::Zm010(_)
                | Self::Zm069(_)
                | Self::Zm016(_)
                | Self::Zm068(_)
                | Self::Zf083(_)
                | Self::Zf093(_)
                | Self::Zf006(_)
                | Self::Zf026(_)
                | Self::Zm053(_)
                | Self::Zm064(_)
                | Self::AfSol(_)
                | Self::Zf042(_)
                | Self::Zf084(_)
                | Self::Zf073(_)
                | Self::Zf067(_)
                | Self::Zm025(_)
                | Self::Zm020(_)
                | Self::Zm050(_)
                | Self::Zf070(_)
                | Self::Zf002(_)
                | Self::Zf032(_)
                | Self::Zm091(_)
                | Self::Zm066(_)
                | Self::Zm089(_)
                | Self::Zm034(_)
                | Self::Zm100(_)
                | Self::Zf086(_)
                | Self::Zf040(_)
                | Self::Zm011(_)
                | Self::Zm098(_)
                | Self::Zm015(_)
                | Self::Zf051(_)
                | Self::Zm065(_)
                | Self::Zf076(_)
                | Self::Zf036(_)
                | Self::Zm033(_)
                | Self::Zf018(_)
                | Self::Zf017(_)
                | Self::Zf049(_)
                | Self::AfMaple(_)
                | Self::Zm082(_)
                | Self::Zm057(_)
                | Self::Zf079(_)
                | Self::Zf022(_)
                | Self::Zm063(_)
                | Self::Zf060(_)
                | Self::Zf019(_)
                | Self::Zm097(_)
                | Self::Zm096(_)
                | Self::Zf023(_)
                | Self::Zf027(_)
                | Self::Zf085(_)
                | Self::Zf077(_)
                | Self::Zm035(_)
                | Self::Zf088(_)
                | Self::Zf024(_)
                | Self::Zf072(_)
                | Self::Zm055(_)
                | Self::Zm052(_)
                | Self::Zf071(_)
                | Self::Zm061(_)
                | Self::Zf078(_)
                | Self::Zm013(_)
                | Self::Zm081(_)
                | Self::Zm037(_)
                | Self::Zf090(_)
                | Self::Zf043(_)
                | Self::Zm058(_)
                | Self::Zm012(_)
                | Self::Zm045(_)
                | Self::Zf075(_)
        )
    }

    pub(super) fn get_speed_v10(&self) -> Result<f32, KokoroError> {
        match self {
            Self::ZmYunyang(v)
            | Self::ZfXiaoni(v)
            | Self::AfJessica(v)
            | Self::BfLily(v)
            | Self::ZfXiaobei(v)
            | Self::ZmYunxia(v)
            | Self::AfHeart(v)
            | Self::BfEmma(v)
            | Self::AmPuck(v)
            | Self::BfAlice(v)
            | Self::HfAlpha(v)
            | Self::BfIsabella(v)
            | Self::AfNova(v)
            | Self::AmFenrir(v)
            | Self::EmAlex(v)
            | Self::ImNicola(v)
            | Self::PmAlex(v)
            | Self::AfAlloy(v)
            | Self::ZmYunxi(v)
            | Self::AfSarah(v)
            | Self::JfNezumi(v)
            | Self::BmDaniel(v)
            | Self::JfTebukuro(v)
            | Self::JfAlpha(v)
            | Self::JmKumo(v)
            | Self::EmSanta(v)
            | Self::AmLiam(v)
            | Self::AmSanta(v)
            | Self::AmEric(v)
            | Self::BmFable(v)
            | Self::AfBella(v)
            | Self::BmLewis(v)
            | Self::PfDora(v)
            | Self::AfNicole(v)
            | Self::BmGeorge(v)
            | Self::AmOnyx(v)
            | Self::HmPsi(v)
            | Self::HfBeta(v)
            | Self::HmOmega(v)
            | Self::ZfXiaoxiao(v)
            | Self::FfSiwis(v)
            | Self::EfDora(v)
            | Self::AfAoede(v)
            | Self::AmEcho(v)
            | Self::AmMichael(v)
            | Self::AfKore(v)
            | Self::ZfXiaoyi(v)
            | Self::JfGongitsune(v)
            | Self::AmAdam(v)
            | Self::IfSara(v)
            | Self::AfSky(v)
            | Self::PmSanta(v)
            | Self::AfRiver(v)
            | Self::ZmYunjian(v) => Ok(*v),
            _ => Err(KokoroError::VoiceVersionInvalid(
                "Expect version 1.0".to_owned(),
            )),
        }
    }

    pub(super) fn get_speed_v11(&self) -> Result<i32, KokoroError> {
        match self {
            Self::Zm029(v)
            | Self::Zf048(v)
            | Self::Zf008(v)
            | Self::Zm014(v)
            | Self::Zf003(v)
            | Self::Zf047(v)
            | Self::Zm080(v)
            | Self::Zf094(v)
            | Self::Zf046(v)
            | Self::Zm054(v)
            | Self::Zf001(v)
            | Self::Zm062(v)
            | Self::BfVale(v)
            | Self::Zf044(v)
            | Self::Zf005(v)
            | Self::Zf028(v)
            | Self::Zf059(v)
            | Self::Zm030(v)
            | Self::Zf074(v)
            | Self::Zm009(v)
            | Self::Zf004(v)
            | Self::Zf021(v)
            | Self::Zm095(v)
            | Self::Zm041(v)
            | Self::Zf087(v)
            | Self::Zf039(v)
            | Self::Zm031(v)
            | Self::Zf007(v)
            | Self::Zf038(v)
            | Self::Zf092(v)
            | Self::Zm056(v)
            | Self::Zf099(v)
            | Self::Zm010(v)
            | Self::Zm069(v)
            | Self::Zm016(v)
            | Self::Zm068(v)
            | Self::Zf083(v)
            | Self::Zf093(v)
            | Self::Zf006(v)
            | Self::Zf026(v)
            | Self::Zm053(v)
            | Self::Zm064(v)
            | Self::AfSol(v)
            | Self::Zf042(v)
            | Self::Zf084(v)
            | Self::Zf073(v)
            | Self::Zf067(v)
            | Self::Zm025(v)
            | Self::Zm020(v)
            | Self::Zm050(v)
            | Self::Zf070(v)
            | Self::Zf002(v)
            | Self::Zf032(v)
            | Self::Zm091(v)
            | Self::Zm066(v)
            | Self::Zm089(v)
            | Self::Zm034(v)
            | Self::Zm100(v)
            | Self::Zf086(v)
            | Self::Zf040(v)
            | Self::Zm011(v)
            | Self::Zm098(v)
            | Self::Zm015(v)
            | Self::Zf051(v)
            | Self::Zm065(v)
            | Self::Zf076(v)
            | Self::Zf036(v)
            | Self::Zm033(v)
            | Self::Zf018(v)
            | Self::Zf017(v)
            | Self::Zf049(v)
            | Self::AfMaple(v)
            | Self::Zm082(v)
            | Self::Zm057(v)
            | Self::Zf079(v)
            | Self::Zf022(v)
            | Self::Zm063(v)
            | Self::Zf060(v)
            | Self::Zf019(v)
            | Self::Zm097(v)
            | Self::Zm096(v)
            | Self::Zf023(v)
            | Self::Zf027(v)
            | Self::Zf085(v)
            | Self::Zf077(v)
            | Self::Zm035(v)
            | Self::Zf088(v)
            | Self::Zf024(v)
            | Self::Zf072(v)
            | Self::Zm055(v)
            | Self::Zm052(v)
            | Self::Zf071(v)
            | Self::Zm061(v)
            | Self::Zf078(v)
            | Self::Zm013(v)
            | Self::Zm081(v)
            | Self::Zm037(v)
            | Self::Zf090(v)
            | Self::Zf043(v)
            | Self::Zm058(v)
            | Self::Zm012(v)
            | Self::Zm045(v)
            | Self::Zf075(v) => Ok(*v),
            _ => Err(KokoroError::VoiceVersionInvalid(
                "Expect version 1.1".to_owned(),
            )),
        }
    }
}
