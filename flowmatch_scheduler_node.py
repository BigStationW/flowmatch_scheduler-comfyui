import torch
from .flowmatch_scheduler import FlowMatchScheduler


class CMFlowMatchSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 10.0}),
                "denoising_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "sigma_max": ("FLOAT", {"default": 1.0}),
                "sigma_min": ("FLOAT", {"default": 0.003 / 1.002}),
                "inverse_timesteps": ("BOOLEAN", {"default": False}),
                "extra_one_step": ("BOOLEAN", {"default": False}),
                "reverse_sigmas": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "create_sigmas"
    CATEGORY = "schedulers"

    def create_sigmas(self, MODEL, num_inference_steps, denoising_strength, shift,
                      sigma_max, sigma_min, inverse_timesteps, extra_one_step, reverse_sigmas):
        scheduler = FlowMatchScheduler(
            num_inference_steps=num_inference_steps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
        )
        # Apply denoising_strength to alter the sigmas
        scheduler.set_timesteps(num_inference_steps, denoising_strength)
        return (scheduler.sigmas,)


NODE_CLASS_MAPPINGS = {
    "FlowMatchSigmas": CMFlowMatchSigmas
}
