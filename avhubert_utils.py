import sys
sys.path.append('./model/av_hubert/fairseq')
from preprocessing import audio_utils as audio
import torch
from torch.utils import data as data_utils
from fairseq.data import data_utils
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from torchvision import transforms
import torch
import cv2
import random

def build_encoder(hubert_root, cfg):
    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
    arg_overrides = {
        "dropout": cfg.dropout,
        "activation_dropout": cfg.activation_dropout,
        "dropout_input": cfg.dropout_input,
        "attention_dropout": cfg.attention_dropout,
        "mask_length": cfg.mask_length,
        "mask_prob": cfg.mask_prob,
        "mask_selection": cfg.mask_selection,
        "mask_other": cfg.mask_other,
        "no_mask_overlap": cfg.no_mask_overlap,
        "mask_channel_length": cfg.mask_channel_length,
        "mask_channel_prob": cfg.mask_channel_prob,
        "mask_channel_selection": cfg.mask_channel_selection,
        "mask_channel_other": cfg.mask_channel_other,
        "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        "encoder_layerdrop": cfg.layerdrop,
        "feature_grad_mult": cfg.feature_grad_mult,
    }
    if cfg.w2v_args is None:
        state = checkpoint_utils.load_checkpoint_to_cpu(
            cfg.w2v_path, arg_overrides
        )
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args
    else:
        state = None
        w2v_args = cfg.w2v_args
        if isinstance(w2v_args, Namespace):
            cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                w2v_args
            )

    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)
    if state is not None:
        task_pretrain.load_state_dict(state['task_state'])
    # task_pretrain.state = task.state

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder


def get_avhubert(hubert_root, ckptpath):

    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_pretraining import LabelEncoderS2SToken
    from fairseq.dataclass.utils import DictConfig

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckptpath])
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.report_accuracy = True

    dictionaries = [task.target_dictionary]
    bpe_tokenizer = task.s2s_tokenizer
    procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
    extra_gen_cls_kwargs = {
        "lm_model": None,
        "lm_weight": 0.0,
    }
    arg_gen = DictConfig({'_name': None, 'beam': 50, 'nbest': 1, 'max_len_a': 1.0, 'max_len_b': 0, 'min_len': 1,
                          'match_source_len': False, 'unnormalized': False, 'no_early_stop': False,
                          'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None,
                          'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0,
                          'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None,
                          'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5,
                          'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None,
                          'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10,
                          'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1,
                          'iter_decode_with_external_reranker': False, 'retain_iter_history': False,
                          'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None,
                          'no_seed_provided': False})
    generator = task.build_generator(
        models, arg_gen, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )
    encoder = build_encoder(hubert_root, saved_cfg.model)
    model_dict_avhubert = models[0].state_dict()
    model_dict_encoder = encoder.state_dict()
    for key in model_dict_encoder.keys():
        model_dict_encoder[key] = model_dict_avhubert['encoder.'+key]
    encoder.load_state_dict(model_dict_encoder)
    return models[0], procs[0], generator, criterion, encoder


def retrieve_avhubert(hubert_root, hubert_ckpt, device):
    avhubert, label_proc, generator, criterion, encoder = get_avhubert(hubert_root, hubert_ckpt)
    """Base configuration"""
    ftlayers = list(range(9, 12))

    ftlayers_full = ['w2v_model.encoder.layers.'+str(layer) for layer in ftlayers]
    for name, p in encoder.named_parameters():
        ft_ind = False
        for layer in ftlayers_full:
            if layer in name:
                ft_ind = True
                break
        if ft_ind:
            p.requires_grad = True
        else:
            p.requires_grad = False

    for p in avhubert.parameters():
        p.requires_grad = False
    avhubert = avhubert.to(device)
    avhubert.eval()
    return avhubert, label_proc, generator, criterion, encoder

def collate_fn(dataBatch):
    """
    Args:
        dataBatch:

    Returns:
        inpBatch: input T_sum*6*96*96, concatenation of all video chips in the time dimension
        gtBatch: output T_sum*3*96*96
        inputLenBatch: bs
        audioBatch: bs*104*T'
        audio_idx: T_sum
        targetBatch: words for lip-reading expert
        padding_mask: bs*T'
        pickedimg: a list of bs elements, each contain some picked indices
        videoBatch: a list of bs elements, each cotain a video
        bbxs: a list of bs elements
    """

    inpBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    gtBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[4] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))
    audio_idx = torch.cat([data[5] + audioBatch.shape[2] * i for i, data in enumerate(dataBatch)], dim=0)

    targetBatch = collater_label([[data[3] for data in dataBatch]])

    bbxs = [data[7] for data in dataBatch]
    pickedimg = [data[5] for data in dataBatch]
    videoBatch = [data[6] for data in dataBatch]

    return inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs

def collater_seq_label_s2s(targets):
    lengths = torch.LongTensor([len(t) for t in targets])
    ntokens = lengths.sum().item()
    pad, eos = 1, 2
    targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
    prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
    return (targets_, prev_output_tokens), lengths, ntokens


def collater_label(targets_by_label):
    targets_list, lengths_list, ntokens_list = [], [], []
    itr = zip(targets_by_label, [-1], [1])
    for targets, label_rate, pad in itr:
        if label_rate == -1:
            targets, lengths, ntokens = collater_seq_label_s2s(targets)
        targets_list.append(targets)
        lengths_list.append(lengths)
        ntokens_list.append(ntokens)
    return targets_list[0], lengths_list[0], ntokens_list[0]


def collater_audio(audios, audio_size):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
    padding_mask = (
        torch.BoolTensor(len(audios), audio_size).fill_(False) #
    )
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            import sys
            sys.exit('Audio segment is longer than the loggest')
    if len(audios[0].shape) == 2:
        collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
    else:
        collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_audios, padding_mask


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


transform = Compose([
    Normalize(0.0, 255.0),
    CenterCrop((88, 88)),
    Normalize(0.421, 0.165)])


def rgb2gray(g, dim):
    glist = g.split([1,1,1], dim=dim)
    return 0.299 * glist[2] + 0.587 * glist[1] + 0.114 * glist[0]


def affine_trans(imgs, video_size):
    h, w, _ = imgs[0][0].shape
    videoSeq = list()
    for i, img in enumerate(imgs):
        new_images = list()
        for j, frame in enumerate(img):
            frame = rgb2gray(frame, 2).squeeze(dim=-1)
            new_images.append(frame)
        new_images = torch.stack(new_images, dim=0)
        videoSeq.append(transform(new_images).unsqueeze(dim=-1))
    collated_videos, padding_mask = collater_audio(videoSeq, video_size)
    return collated_videos


def emb_roi2im(pickedimg, imgs, bbxs, pre, device):
    trackid = 0
    height, width, _ = imgs[0][0].shape
    for i in range(len(pickedimg)):
        idimg = pickedimg[i]
        imgs[i] = imgs[i].float().to(device)
        for j in range(len(idimg)):
            bbx = bbxs[i][idimg[j]]
            if bbx[2] > width: bbx[2] = width
            if bbx[3] > height: bbx[3] = height
            resize2ori = transforms.Resize([bbx[3] - bbx[1], bbx[2] - bbx[0]])
            try:
                resized = resize2ori(pre[trackid + j] * 255.).permute(1, 2, 0)
                imgs[i][idimg[j]][bbx[1]:bbx[3], bbx[0]:bbx[2], :] = resized
            except:
                print(bbx, resized.shape)
                import sys
                sys.exit()
        trackid += len(idimg)
    return imgs


def images2avhubert(pickedimg, imgs, bbxs, pre, video_size, device):
    imgs = emb_roi2im(pickedimg, imgs, bbxs, pre, device)
    processed_img = affine_trans(imgs, video_size).to(device)
    return processed_img