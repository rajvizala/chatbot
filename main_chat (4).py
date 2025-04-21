


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================
#     Non-Interactive main_chat.py (Single Turn)
# =============================================
# This script is modified to accept a --question argument
# and run for a single turn instead of looping interactively.

import argparse
import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, BitsAndBytesConfig # Keep import for structure, but disable usage with device_map
# from accelerate import Accelerator # No longer explicitly needed if just using device_map="auto"
# from accelerate.utils import infer_auto_device_map # Can be used for more control if needed

# Ensure model imports point to the correct relative path if running from /kaggle/working/PoseGPT/
# If PoseGPT is in /kaggle/working/, these paths should be fine.
# Adjust if your structure is different (e.g., sys.path.append might be needed)
try:
    from model.chatpose import ChatPoseForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure you are running this script from the '/kaggle/working/PoseGPT' directory")
    print("or that the necessary paths are in sys.path.")
    sys.exit(1)


import json
from tqdm import tqdm
from io import BytesIO
from transformers import TextStreamer
import requests # Assuming load_image might be used later or for web paths
from PIL import Image # Assuming load_image might be used later
import matplotlib # Use Agg backend for non-interactive saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- NOTES ---
# Uses HF Accelerate's device_map="auto".
# 4-bit/8-bit quantization is NOT supported with device_map='auto' here.
# DeepSpeed removed.
# SMPL CUDA rasterizer compilation might still occur/fail depending on env.
# --- /NOTES ---

def parse_args(args):
    parser = argparse.ArgumentParser(description="ChatPose chat (Non-Interactive Single Turn)")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--version", default="YaoFeng/CHATPOSE-V0", help="Model name or path")
    parser.add_argument("--image_file", type=str, default=None, help="Path to the input image file")
    parser.add_argument("--image_dir", default="./dataset/Yoga-82") # Unused in chat
    parser.add_argument("--json_path", default="./dataset/Yoga-82/yoga_dataset.json") # Unused in chat
    parser.add_argument("--vis_save_path", default="./vis_output", type=str, help="Directory to save visualizations")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference (torch_dtype)",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size (unused?)")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int) # For loading compatibility
    parser.add_argument("--out_dim", default=144, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="8-bit quantization (Not supported with device_map='auto' in this script)")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="4-bit quantization (Not supported with device_map='auto' in this script)")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # Arguments likely related to model config, keep them
    parser.add_argument("--text_embeddings_for_global", action="store_true", default=False)
    parser.add_argument("--predict_global_orient", action="store_true", default=False)
    parser.add_argument("--cat_image_embeds", action="store_true", default=False)
    # --- ADDED FOR NON-INTERACTIVE USE ---
    parser.add_argument("--question", type=str, required=True, help="The question to ask the model non-interactively")
    # ---

    return parser.parse_args(args)

# Keep original visualize_LLM, load_image, preprocess functions
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def visualize_LLM(save_path, questions, answers, visualizations):
    # Ensure numpy is imported
    import numpy as np

    # Format questions and answers (ensure f-string syntax is correct)
    formatted_questions = [f'Q{i+1}: {q}' for i, q in enumerate(questions)]
    formatted_answers = [f'A{i+1}: {a.replace("[SEG] ", "[POSE]").replace("[SEG]", "[POSE]")}' for i, a in enumerate(answers)]

    # Handle cases where lengths might mismatch slightly if generation failed etc.
    num_items = max(len(formatted_questions), len(formatted_answers), len(visualizations))
    vis_to_plot = visualizations + [None] * (num_items - len(visualizations)) # Pad viz list if needed

    fig, ax = plt.subplots(nrows=num_items, ncols=1, figsize=(12, 6 * num_items), squeeze=False) # Always return 2D array

    for k in range(num_items):
        ax_ = ax[k, 0] # Access subplot correctly
        if k < len(vis_to_plot) and vis_to_plot[k] is not None:
            try:
                # Assuming visualization is a tensor needing CPU, numpy conversion
                image_data = vis_to_plot[k].cpu().float().numpy() # Use float before potential transpose
                # Handle potential channel permutations (e.g., CHW -> HWC)
                if image_data.ndim == 3 and image_data.shape[0] in [1, 3]: # Check for channel dim
                    if image_data.shape[0] == 1: # Grayscale C=1, H, W -> H, W
                         image_data = image_data.squeeze(0)
                    else: # C=3, H, W -> H, W, C
                         image_data = np.transpose(image_data, (1, 2, 0))

                # Handle normalization (Clip just in case, then scale)
                image_data = np.clip(image_data, 0, 1) # Assuming range was [0, 1]
                image_data = (image_data * 255).astype(np.uint8)

                ax_.imshow(image_data)
            except Exception as e:
                print(f"Error displaying visualization {k}: {e}")
                ax_.text(0.5, 0.5, 'Viz Error', horizontalalignment='center', verticalalignment='center')
        else:
           # Keep placeholder if no visualization or error
           ax_.text(0.5, 0.5, 'No Viz', horizontalalignment='center', verticalalignment='center')

        ax_.axis('off')
        q_text = formatted_questions[k] if k < len(formatted_questions) else 'Q: N/A'
        a_text = formatted_answers[k] if k < len(formatted_answers) else 'A: N/A'
        ax_.set_title(q_text + '
' + a_text, wrap=True, color='g', fontsize=10) # Smaller font

    plt.tight_layout() # Adjust layout
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150) # Adjust padding/dpi
    plt.close(fig) # Close the figure to free memory
    print(f"Visualization saved to {save_path}")


def main(args):
    args = parse_args(args)

    
    # --- Setup Save Paths ---
    if args.exp_name is not None:
        save_name = args.exp_name.upper()
        # If version is explicitly set, maybe don't override it? Or decide precedence.
        # args.version = f"./checkpoints/{save_name}" # Let's assume args.version is the primary identifier
        args.vis_save_path = os.path.join("./vis_output", save_name) # Use join for paths
    else:
        # Try to get a meaningful name from version path/name
        save_name = args.version.split('/')[-1] if '/' in args.version else args.version
        args.vis_save_path = os.path.join("./vis_output", save_name)
    os.makedirs(args.vis_save_path, exist_ok=True)
    print(f"Saving visualizations to: {args.vis_save_path}")

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.version,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    except Exception as e:
        print(f"Error loading tokenizer for {args.version}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Determine Torch Dtype ---
    torch_dtype = torch.float32
    quantization_config = None # Initialize

    if args.load_in_4bit:
        print("Setting up 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bf16 if possible
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # llm_int8_skip_modules=["visual_model"], # Check if skipping vision model is still needed/desired
        )
        torch_dtype = None # Let BnB handle dtype for quantized parts
        print("Using 4-bit quantization.")

    elif args.load_in_8bit:
        print("Setting up 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_skip_modules=["visual_model"], # Check if skipping vision model is still needed/desired
        )
        torch_dtype = None # Let BnB handle dtype for quantized parts
        print("Using 8-bit quantization.")
    if args.load_in_4bit or args.load_in_8bit:
        compute_dtype = torch.float16

    # torch_dtype = torch.float32
    elif args.precision == "bf16":
        if torch.cuda.is_bf16_supported():
             torch_dtype = torch.bfloat16
             print("Using bfloat16 precision.")
        else:
             print("Warning: bfloat16 not supported, falling back to float16.")
             torch_dtype = torch.float16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        print("Using float16 precision.")
    else:
        print("Using float32 precision.")

    # --- Load Model ---
    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update({"out_dim": args.out_dim})
    kwargs.update({"text_embeddings_for_global": args.text_embeddings_for_global})
    kwargs.update({"predict_global_orient": args.predict_global_orient})
    kwargs.update({"cat_image_embeds": args.cat_image_embeds})

    print(f"Loading model {args.version} with device_map='auto' and dtype {torch_dtype}...")
    try:
        model = ChatPoseForCausalLM.from_pretrained(
            args.version,
            low_cpu_mem_usage=True,
            vision_tower=args.vision_tower,
            seg_token_idx=args.seg_token_idx,   
            quantization_config=quantization_config,
            # device_map="auto", # Key for multi-GPU / accelerate integration
            **kwargs
        )
    except Exception as e:
         print(f"Error loading model {args.version}: {e}", file=sys.stderr)
         print("This might be due to incorrect model path, network issues, or CUDA setup problems (check nvidia-smi).")
         sys.exit(1)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Setup Vision ---
    try:
        vision_tower = model.get_model().get_vision_tower()
        if hasattr(vision_tower, 'load_model') and not vision_tower.is_loaded:
            print("Loading vision tower weights...")
            vision_tower.load_model()
        clip_image_processor = vision_tower.image_processor
    except Exception as e:
         print(f"Error setting up vision tower: {e}", file=sys.stderr)
         # Decide if this is fatal or if text-only can proceed
         if args.image_file:
             print("Vision tower error is fatal when an image is provided.", file=sys.stderr)
             sys.exit(1)
         else:
              print("Proceeding in text-only mode due to vision tower error.", file=sys.stderr)
              clip_image_processor = None # Ensure it's None if failed
    if not quantization_config:
        print("Moving model to cuda:0")
        try:
            model.to('cuda:0') # Move main model parts
            vision_tower.to('cuda:0')# Move vision tower
            input_device = torch.device("cuda:0")
        except Exception as e:
            print(f"Error moving model to GPU: {e}")
            sys.exit(1)
    else:
        # Quantization with bitsandbytes usually handles device placement to cuda:0 by default
        print("Model loaded with quantization (device placement handled by BitsAndBytes).")
        # Determine input device (likely cuda:0 where the quantized model lives)
        try:
            input_device = model.get_input_embeddings().weight.device
        except Exception:
            input_device = torch.device("cuda:0") # Assume cuda:0

    print(f"Model ready on device: {input_device}")

    model.eval()
    print(f"Model loaded successfully on device map: {model.hf_device_map}")
    try:
         input_device = model.get_input_embeddings().weight.device
    except Exception as e:
         print(f"Warning: Could not determine input device from embeddings ({e}). Using default cuda:0 or cpu.", file=sys.stderr)
         input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Placing inputs on device: {input_device}")


    # --- Load and Process Image (if provided) ---
    image_path = args.image_file
    image_clip = None
    image = None
    imagename = 'textonly.png' # Default name if no image
    pad_image = False

    if image_path:
        if os.path.exists(image_path):
            print(f"Loading image from {image_path}")
            try:
                image_pil = load_image(image_path) # Use PIL loader
                image_np = np.array(image_pil) # Convert to numpy for CLIP proc
                # original_size_list = [image_np.shape[:2]] # Keep if needed

                # Preprocess image for CLIP vision tower
                if clip_image_processor:
                    image_clip_processed = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
                    # image_clip = image_clip_processed.unsqueeze(0).to(input_device, dtype=torch_dtype)
                    image_clip = image_clip_processed.unsqueeze(0).to(input_device, dtype=compute_dtype)
                    # Create the smaller image version (256x256) for pose head?
                    image_interp = F.interpolate(image_clip.float(), size=[256, 256], mode='bilinear', align_corners=False)
                    image = image_interp.to(dtype=compute_dtype) # Keep on same device

                    pad_image = True # Flag to add image token
                    imagename = os.path.basename(image_path)
                else:
                    print("Warning: Image provided, but CLIP processor is not available. Cannot process image.", file=sys.stderr)
                    pad_image = False # Cannot use image

            except Exception as e:
                 print(f"Error loading or processing image {image_path}: {e}", file=sys.stderr)
                 print("Proceeding without image.", file=sys.stderr)
                 pad_image = False
        else:
            print(f"Warning: Image file specified but not found: {image_path}", file=sys.stderr)
            print("Proceeding without image.", file=sys.stderr)
            pad_image = False
    else:
        print("No image file provided. Running in text-only mode.")
        pad_image = False


    # --- Setup Conversation ---
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    roles = conv.roles

    questions = []
    answers = []
    visualizations = []
    pred_smpl_params = None # Initialize

    # --- Start Single Turn Execution Block ---
    if not args.question:
        print('Error: --question argument is required for non-interactive mode.', file=sys.stderr)
        # No need to exit here, argparse required=True should handle it, but defensive check is ok
        inp = "Default question: Describe the content." # Or provide a default
    else:
        inp = args.question

    print(f'{roles[0]}: {inp}') # Echo the question
    print(f"{roles[1]}: ", end="", flush=True) # Print bot role prompt, flush ensures it shows before generation

    questions.append(inp)
    current_prompt = inp

    if pad_image:
        # Prepend image token only on the first turn if an image exists
        current_prompt = DEFAULT_IMAGE_TOKEN + "
" + inp # Use escaped newline for string
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            current_prompt = current_prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], current_prompt)
    conv.append_message(conv.roles[1], None) # Placeholder for bot response
    prompt = conv.get_prompt()
    # pad_image = False # Not strictly needed anymore as loop is gone

    try:
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        # Move input_ids to the same device as the model's input embeddings
        input_ids = input_ids.unsqueeze(0).to(input_device)
    except Exception as e:
         print(f"Error tokenizing prompt: {e}", file=sys.stderr)
         sys.exit(1)


    # --- Generate Response ---
    try:
        with torch.inference_mode():
             output_ids, predictions, pred_smpl_params = model.evaluate(
                 image_clip, # Can be None
                 image,      # Can be None
                 input_ids,
                 max_new_tokens=args.model_max_length, # Use arg
                 tokenizer=tokenizer,
                 return_smpl=True,
             )

        # Ensure output_ids are on CPU for decoding
        # Handle potential empty output or only prompt being returned
        if output_ids.shape[1] > input_ids.shape[1]:
             output_ids_decoded = output_ids[0, input_ids.shape[1]:].cpu()
        else:
             output_ids_decoded = torch.tensor([], dtype=torch.long) # Empty tensor if no new tokens

        text_output = tokenizer.decode(output_ids_decoded, skip_special_tokens=True).strip() # Use skip_special_tokens=True commonly
        # Compatibility replace
        text_output = text_output.replace("[SEG]", "[POSE]") # Handle old token if needed
        print(text_output) # Print the final decoded output

        conv.messages[-1][-1] = text_output # Store final answer in conversation
        answers.append(text_output)
        visualizations.append(predictions) # Store predictions tensor (might be on GPU)

    except Exception as e:
         print(f"Error during model generation or evaluation: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc() # Print full traceback
         # Continue to try saving what we have if possible

    # --- End Single Turn Execution Block ---


    # --- Visualization and Saving ---
    # Ensure save path is based on actual imagename used
    save_path_base = os.path.join(args.vis_save_path, os.path.splitext(imagename)[0]) # Use filename without ext

    if questions or answers or visualizations: # Only visualize if there's something to show
        viz_save_path = save_path_base + "_qa_viz.png"
        try:
            visualize_LLM(viz_save_path, questions, answers, visualizations)
        except Exception as e:
            print(f"Could not save visualization to {viz_save_path}: {e}", file=sys.stderr)
            print("This might be due to the SMPL renderer needing CUDA, matplotlib issues, or other errors.", file=sys.stderr)
    else:
        print("No questions/answers/visualizations generated to save.")


    if pred_smpl_params is not None:
        # Ensure params are on CPU before saving
        pred_smpl_params_cpu = {}
        try:
            for k, v in pred_smpl_params.items():
                 if isinstance(v, torch.Tensor):
                     pred_smpl_params_cpu[k] = v.cpu()
                 else:
                     pred_smpl_params_cpu[k] = v # Keep non-tensors as is (like faces)

            pkl_path = save_path_base + "_smpl.pkl"
            try:
                 with open(pkl_path, 'wb') as f:
                     import pickle
                     pickle.dump(pred_smpl_params_cpu, f)
                 print(f"SMPL parameters saved as {pkl_path}")
            except Exception as e:
                 print(f"Error saving SMPL parameters to {pkl_path}: {e}", file=sys.stderr)


            # Save OBJ file (assuming util is available)
            try:
                 # Check if vertices/faces exist before proceeding
                 if 'vertices' in pred_smpl_params_cpu and 'faces' in pred_smpl_params_cpu:
                     from model.smpl.util import write_obj # Keep import local if util might be missing
                     objpath = save_path_base + "_smpl.obj"
                     # Ensure vertices/faces are numpy arrays on CPU
                     vertices_np = pred_smpl_params_cpu['vertices'].float().numpy().squeeze()
                     faces_np = pred_smpl_params_cpu['faces']
                     if isinstance(faces_np, torch.Tensor): # Faces might be tensor or numpy already
                         faces_np = faces_np.cpu().numpy()

                     write_obj(objpath, vertices_np, faces_np)
                     print(f"SMPL mesh saved as {objpath}")
                 else:
                      print("Skipping OBJ save: Missing 'vertices' or 'faces' in smpl params.")
            except ImportError:
                print("Skipping OBJ save: model.smpl.util.write_obj not found.", file=sys.stderr)
            except KeyError as e:
                print(f"Skipping OBJ save: Missing key {e} in pred_smpl_params.", file=sys.stderr)
            except Exception as e:
                print(f"Error saving SMPL mesh to {objpath}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing SMPL parameters for saving: {e}", file=sys.stderr)
    else:
        print("No SMPL parameters generated to save.")


if __name__ == "__main__":
    # Check if running in an environment where sys.argv might be manipulated (like some notebooks)
    if len(sys.argv) > 1 and sys.argv[1].startswith("ipykernel_launcher"):
         print("Running in IPython/Jupyter environment, attempting to parse default args.")
         # Create dummy args or use defaults if applicable, as sys.argv is different
         # This part is tricky and might need specific adjustments based on how you run it
         # For command-line execution, sys.argv[1:] is correct.
         # For !python execution in notebook, it's also usually correct.
         main(sys.argv[1:])
    else:
         # Standard execution
         main(sys.argv[1:])



