import torch
from torch import nn
from torch.nn import functional as F

class Talking_Face_Generator(nn.Module):
    def __init__(self):
        super(Talking_Face_Generator, self).__init__()

        # Encoder blocks
        self.video_encoder_block1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.video_encoder_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self._make_res_block(32, 32, kernel_size=3, stride=1, padding=1),
            self._make_res_block(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.video_encoder_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.video_encoder_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.video_encoder_block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            self._make_res_block(256, 256, kernel_size=3, stride=1, padding=1),
            self._make_res_block(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.video_encoder_block6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            self._make_res_block(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.video_encoder_block7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self._make_res_block(32, 32, kernel_size=3, stride=1, padding=1),
            self._make_res_block(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            self._make_res_block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder blocks
        self.video_decoder_block1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.video_decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            self._make_res_block(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.video_decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            self._make_res_block(512, 512, kernel_size=3, stride=1, padding=1),
            self._make_res_block(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.video_decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            self._make_res_block(384, 384, kernel_size=3, stride=1, padding=1),
            self._make_res_block(384, 384, kernel_size=3, stride=1, padding=1)
        )

        self.video_decoder_block5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            self._make_res_block(256, 256, kernel_size=3, stride=1, padding=1),
            self._make_res_block(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.video_decoder_block6 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1),
            self._make_res_block(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.video_decoder_block7 = nn.Sequential(
            nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1),
            self._make_res_block(64, 64, kernel_size=3, stride=1, padding=1)
        )



        # Output block
        self.output_block = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _make_res_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        block = _ResidualBlock(block)
        return block

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Audio encoding
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        # Face encoding
        face_feats = []
        x = face_sequences
        x = self.video_encoder_block1(x)
        face_feats.append(x)
        x = self.video_encoder_block2(x)
        face_feats.append(x)
        x = self.video_encoder_block3(x)
        face_feats.append(x)
        x = self.video_encoder_block4(x)
        face_feats.append(x)
        x = self.video_encoder_block5(x)
        face_feats.append(x)
        x = self.video_encoder_block6(x)
        face_feats.append(x)
        x = self.video_encoder_block7(x)
        face_feats.append(x)

        # Face decoding
        x = audio_embedding
        x = self.video_decoder_block1(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block2(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block3(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block4(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block5(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block6(x)
        x = torch.cat((x, face_feats.pop()), dim=1)
        x = self.video_decoder_block7(x)
        x = torch.cat((x, face_feats.pop()), dim=1)

        # Output
        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)
        else:
            outputs = x

        return outputs

class _ResidualBlock(nn.Module):
    def __init__(self, block):
        super(_ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x):
        residual = x
        out = self.block(x)
        return out + residual

