_batch_norm = "batch_norm={}".format(dict(momentum=0.01, eps=1e-3))
hparams = dict(
    tag="offsets",
    audio=dict(
        feature=dict(
            # features
            name="mel",
            with_delta=True,
            # input window
            sliding_window_frames=64,
            # scaling
            scaling=1,
            # random augment inputs
            random_noise=0.01,
            random_reverb=False,
            random_preemph=0.95,
            random_pitch_shift=False,
            random_time_stretch=False,
            random_mel_extra=[5, 4],
            random_mel_noise=None,
            random_mel_scale=0.15,
            random_mel_dropout=0.15,
            random_mel_tremolo=None,
        )
    ),
    anime=dict(),
    dataset_speech=None,
    dataset_anime=dict(
        type="voca",
        name="voca-offsets",
        train_list=["train.csv"],
        valid_list=["valid.csv"],
    ),
    optim=dict(
        name="Adam",
        args=dict(lr=1e-4, weight_decay=0, __entirety__=True),
        lr_scheduler=None
    ),
    loss=dict(
        __entirety__=True,
        ploss_scale=1,
        mloss_scale=1,
        eloss_scale=1,
        dynamic_scalar=True,
        phoneme_cross_entropy_weight=False,
        anime_loss_weight=[None, "anime_weight", "viseme_weight"][0],
    ),
    ensembling_ms=0,
    save_video=True,
    model=dict(
        __entirety__=True,
        # model global settings
        verbose=True,
        weight_norm=True,
        face_data_type="verts_off_3d",
        prediction_type="face_data",
        # main modules
        audio_encoder=dict(
            __entirety__=True,
            layers=[
                ("permute",     (0, 3, 2, 1)),  # N,T,F,C -> N,C,F,T
                ("conv2d",      3,  32, (3, 1), (1, 1), "act=lrelu@a:0.2", _batch_norm), ("pool2d", "max", (2, 1)),
                ("conv2d",      32, 64, (3, 1), (1, 1), "act=lrelu@a:0.2", _batch_norm), ("pool2d", "max", (2, 1)),
                ("conv2d",      64, 64, (1, 1), (1, 1), "act=lrelu@a:0.2", _batch_norm),
                ("freq-lstm",   64, 32, "hidden_size=128", "output_size=256"),
                ("squeeze",     2),  # N,C,T
                ("permute",     (0, 2, 1)),  # N,T,C
                ("lstm",        256, 256, "num_layers=2", "bidirectional=True", "dropout=0.1"),
                ("attn", "bah", 512, 128, 2, "scale_score_at_eval=1.0"),
            ]
        ),
        output=dict(
            __entirety__=True,
            layers=[
                ("fc", 520, 512, "act=lrelu@a:0.2", "cat_condition=2"),
                ("fc", 512, 256, "act=tanh"),
                ("fc", 256, 59,  "act=linear"),
            ],
            output_dim=15069,
            using_pca=True,
            pca_trainable=False,
            pca=("{DATASET_ANIME_ROOT}/pca/compT.npy", "{DATASET_ANIME_ROOT}/pca/means.npy"),
        ),
        # condition: speakers
        speaker_embedding=dict(
            using_onehot=True,
            num_speakers=8,
            # embedding_size=32,
        )
    ),
    trainer=dict(
        # for dataloaders
        anime_loader=dict(
            batch_size=50,
            multiple_workers=True,
        ),
        speech_loader=dict(
            batch_size=10,
            multiple_workers=True,
        ),
        max_epochs=100,
        plot_gap_steps=400,
        eval_gap_epochs=10,
        save_gap_epochs=10,
        reference_metric="ploss",
        reference_metric_larger=False,
        eval_debug=False,
        evaluate=dict(
            test=[
                ("/home/yuki/Videos/obama/test-000.mp4", "speaker=m1"),
                # ("/Users/chaiyujin/Projects/Update/EvalSrcs/noisy@charlie.wav", "speaker=m1"),
                # ("/Users/chaiyujin/Projects/Update/EvalSrcs/noisy@f1_radio.mp4", "speaker=m1"),
                # ("/Users/chaiyujin/Projects/Update/EvalSrcs/noisy@nba_interview.mp4", "speaker=m1"),
                # ("/Users/chaiyujin/Projects/Update/EvalSrcs/noisy@obama_acceptance.mp4", "speaker=m1"),
            ]
        )
    ),
    device="cuda:0",
)
