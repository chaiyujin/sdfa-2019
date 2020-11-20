hparams = dict(
    tag="default",
    audio=dict(),
    anime=dict(),
    dataset_speech=None,
    dataset_anime=dict(),
    optim=dict(
        __entirety__=True,
        name="Adam",
        args=dict(lr=1e-3, weight_decay=0, __entirety__=True),
        lr_scheduler=dict(
            name="NoamDecay",
            args=dict(mode="epoch", warmup_iters=10),
            __entirety__=True,
        ),
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
        face_data_type="dgrad_3d",
        prediction_type="face_data",
        # main modules
        audio_encoder=None, asr_encoder=None,  # encoder alternation
        time_aggregator=None,
        anime_decoder=None,
        # optional
        speaker_embedding=None,    # condition: speakers
        emotion_embedding=None,    # condition: emotions
        vector_quantizer=None,     # quantize: encoded audio features
        phoneme_classifier=None,   # multi-tasks: classifier of phonemes
        audio_reconstructor=None,  # multi-tasks: reconstructor of audio features
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
        max_epochs=150,
        plot_gap_steps=400,
        save_gap_epochs=10,
        eval_gap_epochs=10,
        valid_gap_epochs=0,
        reference_metric="ploss",
        reference_metric_larger=False,
        evaluate=dict(
            test=[
                ("/home/chaiyujin/Videos/eval_data/clips/speech@clip0.mp4", "speaker=m1"),
            ]
        )
    ),
    device="cuda:0",
)
