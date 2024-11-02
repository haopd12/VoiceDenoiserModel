import librosa
from VoiceDenoiser.VoiceDenoiser.df.enhance import *
import torchaudio as ta

class VoiceDenoiser(object):
    def mix_at_snr(self, clean, noise, snr, eps=1e-10):
            """Mix clean and noise signal at a given SNR.
            Args:
                clean: 1D Tensor with the clean signal to mix.
                noise: 1D Tensor of shape.
                snr: Signal to noise ratio.
            Returns:
                clean: 1D Tensor with gain changed according to the snr.
                noise: 1D Tensor with the combined noise channels.
                mix: 1D Tensor with added clean and noise signals.
            """
            clean = torch.as_tensor(clean).mean(0, keepdim=True)
            noise = torch.as_tensor(noise).mean(0, keepdim=True)
            if noise.shape[1] < clean.shape[1]:
                noise = noise.repeat((1, int(math.ceil(clean.shape[1] / noise.shape[1]))))
            max_start = int(noise.shape[1] - clean.shape[1])
            start = torch.randint(0, max_start, ()).item() if max_start > 0 else 0
            logger.debug(f"start: {start}, {clean.shape}")
            noise = noise[:, start : start + clean.shape[1]]
            E_speech = torch.mean(clean.pow(2)) + eps
            E_noise = torch.mean(noise.pow(2))
            K = torch.sqrt((E_noise / E_speech) * 10 ** (snr / 1000) + eps)
            noise = noise / K
            mixture = clean + noise
            logger.debug("mixture: {mixture.shape}")
            assert torch.isfinite(mixture).all()
            max_m = mixture.abs().max()
            if max_m > 1:
                logger.warning(f"Clipping detected during mixing. Reducing gain by {1/max_m}")
                clean, noise, mixture = clean / max_m, noise / max_m, mixture / max_m
            return clean, noise, mixture

    def __init__(self,model_name, atten_lim = None, noisy_fn=None):
        self.model, self.df_state, _, _ = init_df(model_name)  # Load default model
        self.df_sr = ModelParams().sr
        self.atten_lim = atten_lim
        self.noisy_fn = noisy_fn
        # audio, meta = load_audio('../1.wav', df_sr, "cpu")
    def denoise(self, audio, sr):
        if sr is not None and sr != self.df_sr:
            audio = resample(audio, sr, self.df_sr)
        if self.noisy_fn is not None: 
            noise, _ = load_audio(None, self.df_sr, "cpu")
            _, _, audio = mix_at_snr(audio, noise, self.df_sr)
        else:
            noise = None
        enhanced_audio = enhance(self.model, self.df_state, audio, True, atten_lim_db=self.atten_lim)
        audio1 = resample(enhanced_audio.to("cpu"), self.df_sr, sr)
        return audio1, sr

if __name__ == "__main__":
    audio, sr = ta.load('./samples/1.wav')
    denoiser = VoiceDenoiser('./VoiceDenoiser/VoiceDenoiser/df/checkpoints', None, None)
    new_audio, new_sr = denoiser.denoise(audio,sr)