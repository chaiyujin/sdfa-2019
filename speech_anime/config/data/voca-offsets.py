
hparams = dict(
    audio=dict(
        sample_rate=8000,
        lpc=dict(
            __entirety__=True,
            order=32,
            win_size=0.064,
            hop_size=0.008,
            win_fn="hamm",
            preemphasis=0.65
        ),
        mel=dict(
            __entirety__=True,
            n_mels=128,
            win_size=0.064,
            hop_size=0.008,
            win_fn="hamm",
            padding=False,
            fmin=50,
            fmax=3600,
            ref_db=20,
            top_db=80,
            normalize=True,
            clip_normalized=True,
            subtract_mean=False,
            preemphasis=0.65
        ),
        feature=dict()
    ),
    anime=dict(
        fps     = 60,
        feature = dict(
            ts_delta    = 100,
            mask_root   = ""
        )
    ),
    dataset_anime=dict(
        root            = "assets/voca-sr8k/offsets",
        primary_key     = "npy_data_path:path",
        denoise_audio   = False,
        audio_target_db = -24.5,  # make sure same with preprocessed vocaset
        # global settings
        speakers=dict(
            m0=0,  f0=1, m1=2, m2=3,
            f1=4,  m3=5, f2=6, f3=7,
            f4=8,  m4=9,  # valid
            m5=10, f5=11  # test
        ),
        speakers_alias=dict(
            # train
            m0 = "FaceTalk_170728_03272_TA",
            f0 = "FaceTalk_170904_00128_TA",
            m1 = "FaceTalk_170725_00137_TA",
            m2 = "FaceTalk_170915_00223_TA",
            f1 = "FaceTalk_170811_03274_TA",
            m3 = "FaceTalk_170913_03279_TA",
            f2 = "FaceTalk_170904_03276_TA",
            f3 = "FaceTalk_170912_03278_TA",
            # valid
            f4 = "FaceTalk_170811_03275_TA",
            m4 = "FaceTalk_170908_03277_TA",
            # test
            m5 = "FaceTalk_170809_00138_TA",
            f5 = "FaceTalk_170731_00024_TA",
        ),
        emotions=dict(neutral=0),
        # ignored in training.
        ignore=dict()
    )
)
