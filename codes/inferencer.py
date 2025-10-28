# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications:
#   - Added multiple VQA reconstruction versions (ver0/ver0_1/ver1) supporting batch image reconstruction
#   - Implemented aspect ratio preservation in image generation (_calculate_target_size_with_aspect_ratio)
#   - Enhanced context management with device consistency checks
#   - Improved CFG (Classifier-Free Guidance) context handling for VQA+reconstruction tasks
#   - Extended support for multiple image batch reconstruction from single VQA answer

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.unimedvl.qwen2_navit import NaiveCache

# System prompts for think mode
VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in your mind, and then generate the image.
The planning process is enclosed within <think> </think> tags; that is, <think> planning process here </think> image here.
'''


class InterleaveInferencer:
    """Interleave inferencer for text and image mixed input/output with Bagel model."""
    
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def _calculate_target_size_with_aspect_ratio(self, original_width, original_height):
        """Calculate target image size preserving aspect ratio with VAE transform constraints."""
        max_size = self.vae_transform.resize_transform.max_size
        min_size = self.vae_transform.resize_transform.min_size
        stride = self.vae_transform.resize_transform.stride
        max_pixels = self.vae_transform.resize_transform.max_pixels

        def make_divisible(value, stride):
            return max(stride, int(round(value / stride) * stride))
        
        def apply_scale(width, height, scale):
            new_width = round(width * scale)
            new_height = round(height * scale)
            new_width = make_divisible(new_width, stride)
            new_height = make_divisible(new_height, stride)
            return new_width, new_height

        scale = min(max_size / max(original_width, original_height), 1.0)
        scale = max(scale, min_size / min(original_width, original_height))
        new_width, new_height = apply_scale(original_width, original_height, scale)

        if new_width * new_height > max_pixels:
            scale = max_pixels / (new_width * new_height)
            new_width, new_height = apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > max_size:
            scale = max_size / max(new_width, new_height)
            new_width, new_height = apply_scale(new_width, new_height, scale)

        return new_height, new_width

    def init_gen_context(self):
        """Initialize generation context with KV cache."""
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        """Update generation context with text input."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        try:
            model_device = None
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
                if hasattr(self.model.language_model.model, 'embed_tokens'):
                    model_device = next(self.model.language_model.model.embed_tokens.parameters()).device

            if model_device is not None:
                corrected_input = {}
                device_mismatches = 0

                for key, value in generation_input.items():
                    if isinstance(value, torch.Tensor):
                        if value.device != model_device:
                            corrected_input[key] = value.to(model_device)
                            device_mismatches += 1
                        else:
                            corrected_input[key] = value
                    else:
                        corrected_input[key] = value

                if device_mismatches > 0:
                    generation_input = corrected_input

        except Exception as e:
            print(f"⚠️  Device consistency check failed: {e}")

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        """Update generation context with image input."""
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext = None,
        cfg_img_precontext = None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0
    ):
        """Generate image using diffusion model with Classifier-Free Guidance."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        """Decode latent to image."""
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)

        if hasattr(self.vae_model, 'parameters'):
            vae_param = next(self.vae_model.parameters())
            vae_device = vae_param.device
            vae_dtype = vae_param.dtype

            if latent.device != vae_device or latent.dtype != vae_dtype:
                latent = latent.to(device=vae_device, dtype=vae_dtype)

        image = self.vae_model.decode(latent)

        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        """Generate text response."""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)

        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )

        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def interleave_inference_for_vqa_reconstruction_ver1(
        self,
        input_lists: List[Union[str, Image.Image]],
        reconstruct_image: bool = False,
        think: bool = False,
        understanding_output: bool = True, # Primarily for VQA
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: tuple = (1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """Perform VQA and optionally reconstruct multiple images from the answer."""
        output_list = []

        vqa_context = self.init_gen_context()
        vqa_img_context = deepcopy(vqa_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    vqa_context = self.update_context_text(input_term, vqa_context)
                    vqa_img_context = self.update_context_text(input_term, vqa_img_context)
                elif isinstance(input_term, Image.Image):
                    processed_img = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    vqa_context = self.update_context_image(processed_img, vqa_context, vae=True, vit=True)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            vqa_answer = self.gen_text(
                vqa_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n
            )
            output_list.append(vqa_answer)

            if reconstruct_image:
                if not vqa_answer or not vqa_answer.strip():
                    return output_list

                input_images = [item for item in input_lists if isinstance(item, Image.Image)]

                if not input_images:
                    return output_list

                cfg_text_precontext = deepcopy(vqa_context)
                cfg_img_precontext = self.update_context_text(vqa_answer, deepcopy(vqa_img_context))

                full_context = self.update_context_text(vqa_answer, deepcopy(vqa_context))

                for img_idx, original_image in enumerate(input_images):
                    original_width, original_height = original_image.size
                    target_image_shapes = self._calculate_target_size_with_aspect_ratio(original_width, original_height)

                    generated_image = self.gen_image(
                        target_image_shapes,
                        full_context,
                        cfg_text_precontext=cfg_text_precontext,
                        cfg_img_precontext=cfg_img_precontext,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    output_list.append(generated_image)

                    processed_gen_img = self.vae_transform.resize_transform(pil_img2rgb(generated_image))
                    full_context = self.update_context_image(processed_gen_img, full_context, vae=True, vit=False)
                    cfg_text_precontext = self.update_context_image(processed_gen_img, cfg_text_precontext, vae=True, vit=False)

        return output_list


    @torch.no_grad()
    def interleave_inference_for_vqa_reconstruction_ver0_1(
        self,
        input_lists: List[Union[str, Image.Image]],
        reconstruct_image: bool = False,
        think: bool = False,
        understanding_output: bool = True, # Primarily for VQA
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: tuple = (1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """
        Performs VQA and optionally reconstructs images from the answer.
        
        Modified to support multiple image reconstruction when multiple images are provided.
        """
        output_list = []
        
        # --- Part 1: VQA (Text Generation) ---
        vqa_context = self.init_gen_context()
        vqa_img_context = deepcopy(vqa_context) # For CFG, context without image
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Process VQA inputs
            for input_term in input_lists:
                if isinstance(input_term, str):
                    vqa_context = self.update_context_text(input_term, vqa_context)
                    vqa_img_context = self.update_context_text(input_term, vqa_img_context)
                elif isinstance(input_term, Image.Image):
                    processed_img = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    # For VQA+Reconstruction, use both ViT and VAE to align with training.
                    vqa_context = self.update_context_image(processed_img, vqa_context, vae=True, vit=True)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            # Generate the VQA text answer
            vqa_answer = self.gen_text(
                vqa_context, 
                do_sample=do_sample, 
                temperature=text_temperature, 
                max_length=max_think_token_n
            )
            output_list.append(vqa_answer)

            # --- Part 2: Optional Multiple Image Reconstruction ---
            if reconstruct_image:
                if not vqa_answer or not vqa_answer.strip():
                    print("Warning: VQA answer is empty, skipping image reconstruction.")
                    return output_list

                input_images = [item for item in input_lists if isinstance(item, Image.Image)]

                if not input_images:
                    print("Warning: No input images found, skipping image reconstruction.")
                    return output_list

                print(f"Found {len(input_images)} input images, will reconstruct each one based on VQA answer.")

                for img_idx, original_image in enumerate(input_images):
                    print(f"Reconstructing image {img_idx + 1}/{len(input_images)}...")
                    
                    # Use the original image's dimensions to maintain aspect ratio
                    original_width, original_height = original_image.size
                    target_image_shapes = self._calculate_target_size_with_aspect_ratio(original_width, original_height)
                    print(f"Original image {img_idx + 1}: {original_width}x{original_height}, Target size: {target_image_shapes[1]}x{target_image_shapes[0]} (W x H)")

                    gen_context = self.init_gen_context()

                    processed_img = self.vae_transform.resize_transform(pil_img2rgb(original_image))
                    cfg_text_precontext = self.update_context_image(processed_img, deepcopy(gen_context), vae=True, vit=True)

                    full_context = self.update_context_text(vqa_answer, deepcopy(cfg_text_precontext))

                    cfg_img_precontext = self.update_context_text(vqa_answer, deepcopy(gen_context))

                    generated_image = self.gen_image(
                        target_image_shapes,
                        full_context,
                        cfg_text_precontext=cfg_text_precontext,
                        cfg_img_precontext=cfg_img_precontext,
                        cfg_text_scale=7.0,
                        cfg_img_scale=7.0,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    output_list.append(generated_image)

        return output_list

    @torch.no_grad()
    def interleave_inference_for_vqa_reconstruction_ver0(
        self,
        input_lists: List[Union[str, Image.Image]],
        reconstruct_image: bool = False,
        think: bool = False,
        understanding_output: bool = True, # Primarily for VQA
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: tuple = (1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """Perform VQA and optionally reconstruct single image from answer."""
        output_list = []

        vqa_context = self.init_gen_context()
        vqa_img_context = deepcopy(vqa_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    vqa_context = self.update_context_text(input_term, vqa_context)
                    vqa_img_context = self.update_context_text(input_term, vqa_img_context)
                elif isinstance(input_term, Image.Image):
                    processed_img = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    vqa_context = self.update_context_image(processed_img, vqa_context, vae=True, vit=True)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            vqa_answer = self.gen_text(
                vqa_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n
            )
            output_list.append(vqa_answer)

            if reconstruct_image:
                if not vqa_answer or not vqa_answer.strip():
                    return output_list

                original_image = None
                for item in input_lists:
                    if isinstance(item, Image.Image):
                        original_image = item
                        break

                if not original_image:
                    return output_list

                original_width, original_height = original_image.size
                target_image_shapes = self._calculate_target_size_with_aspect_ratio(original_width, original_height)

                gen_context = self.init_gen_context()

                processed_img = self.vae_transform.resize_transform(pil_img2rgb(original_image))
                cfg_text_precontext = self.update_context_image(processed_img, deepcopy(gen_context), vae=True, vit=True)

                full_context = self.update_context_text(vqa_answer, deepcopy(cfg_text_precontext))

                cfg_img_precontext = self.update_context_text(vqa_answer, deepcopy(gen_context))

                generated_image = self.gen_image(
                    target_image_shapes,
                    full_context,
                    cfg_text_precontext=cfg_text_precontext,
                    cfg_img_precontext=cfg_img_precontext,
                    cfg_text_scale=7.0,
                    cfg_img_scale=7.0,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                output_list.append(generated_image)

        return output_list
    
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """
        Main interleaved inference function

        Supports mixed text and image inputs, determines output type based on parameters

        Args:
            input_lists: Input list containing strings and PIL images
            think: Whether to enable think mode (model thinks before answering/generating)
            understanding_output: True=understanding task (text output), False=generation task (image output)
            Other parameters: Various generation hyperparameters

        Returns:
            Output list containing generated text and/or images
        """
        output_list = []

        gen_context = self.init_gen_context()
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    def __call__(
        self,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[str] = None,
        inference_ver = 0,
        **kargs
    ) -> Dict[str, Any]:
        """Main inference entry point."""
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, list):
                input_list.extend(image)
            else:
                input_list.append(image)
        if text is not None:
            input_list.append(text)

        if inference_ver == 0:
            output_list = self.interleave_inference(input_list, **kargs)
        elif inference_ver == 1:
            output_list = self.interleave_inference_for_vqa_reconstruction_ver1(input_list, **kargs)
        else:
            raise ValueError(f"Unsupported inference_ver: {inference_ver}")

        for i in output_list:
            if isinstance(i, Image.Image):
                if output_dict['image'] is None:
                    output_dict['image'] = []
                output_dict['image'].append(i)
            elif isinstance(i, str):
                output_dict['text'] = i

        if isinstance(output_dict['image'], list) and len(output_dict['image']) == 1:
            output_dict['image'] = output_dict['image'][0]

        return output_dict
