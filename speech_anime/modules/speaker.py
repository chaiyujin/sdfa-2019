import saber
import torch


class SpeakerEmbedding(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hp = hparams.model.speaker_embedding
        self.using_onehot = hp.using_onehot
        self.num_speakers = hp.num_speakers
        if not self.using_onehot:
            self._embedding_layer = torch.nn.Embedding(
                num_embeddings = self.num_speakers,
                embedding_dim=hp.embedding_size,
                padding_idx=None
            )
            self.condition_size = hp.embedding_size
        else:
            self.condition_size = hp.num_speakers

    def forward(self, inputs):
        assert inputs.dim() == 1
        if self.using_onehot:
            embeddings = saber.nn.functions.one_hot(inputs, self.num_speakers).float()
        else:
            embeddings = self._embedding_layer(inputs)
        return embeddings
