import uvicorn
from fastapi import APIRouter, FastAPI, Request, requests
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from library.dreambooth_folder_creation_gui import dreambooth_folder_preparation
from typing import List
from modules import errors
from modules.api import models
from threading import Lock
from lora_gui import train_model
import os
import base64
import uuid
import shutil
import boto3
from pydantic import BaseModel
from fastapi import BackgroundTasks

import concurrent.futures

class ImageItem(BaseModel):
    base64_data: str

def api_middleware(app: FastAPI):
    rich_available = True
    try:
        import anyio  # importing just so it can be placed on silent list
        import starlette  # importing just so it can be placed on silent list
        from rich.console import Console
        console = Console()
    except Exception:
        rich_available = False

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        # do not print backtrace on known httpexceptions
        if not isinstance(e, HTTPException):
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[
                                        anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        api_middleware(self.app)

        self.add_api_route("/loraapi/v1/training", self.create_train,
                           methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.fetch_train_log,
                           methods=["GET"], response_model=models.TrainLogResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.check_train,
                           methods=["POST"], response_model=models.TrainInfoResponse)
        self.add_api_route("/loraapi/v1/training/{id}", self.delete_training,
                           methods=["DELETE"], response_model=models.TrainInfoResponse)

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def check_train(self, id: str):
        base_dir = os.path.abspath(os.curdir)
        train_dir = os.path.join(base_dir, 'train_data', id)

        if os.path.exists(train_dir):
            return models.TrainInfoResponse(title="Start LoRA training", description="Train ID exists", info="Success")
        else:
            return models.TrainInfoResponse(title="Start LoRA training", description="Can't find specific train id", info="Failed")

    def fetch_train_log(self, id: str):
        base_dir = os.path.abspath(os.curdir)
        log_dir = os.path.join(base_dir, 'train_data', id, 'log')
        log_file_path = os.path.join(log_dir, 'log.txt')

        if not os.path.isfile(log_file_path):
            raise HTTPException(status_code=404, detail="Log file not found")

        with open(log_file_path, 'r') as log_file:
            train_log = log_file.read()

        return models.TrainLogResponse(title="LoRA training log", description="Log fetched", info=train_log, train_id=id)

    def train_model_bg(self, save_dir, train_data_dir, reg_data_dir, model_path, instance_prompt, class_prompt, max_resolution):
        base_dir = os.path.abspath(os.curdir)

        dreambooth_folder_preparation(
            save_dir,
            40,
            instance_prompt,
            f"{reg_data_dir}/{class_prompt}",
            1,
            class_prompt,
            train_data_dir
        )

        shutil.rmtree(save_dir)

        train_model(
            headless={'label': 'True'},
            print_only={'label': 'False'},
            pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
            v2=False,
            v_parameterization=False,
            sdxl=False,
            logging_dir=f"{train_data_dir}/log",
            train_data_dir=f"{train_data_dir}/img",
            reg_data_dir=f"{train_data_dir}/reg",
            output_dir=f"{train_data_dir}/model",
            max_resolution="512,512",
            learning_rate=0.0001,
            lr_scheduler="cosine",
            lr_warmup=10,
            train_batch_size=1,
            epoch=3,
            save_every_n_epochs=3,
            mixed_precision="fp16",
            save_precision="fp16",
            seed="",
            num_cpu_threads_per_process=2,
            cache_latents=True,
            cache_latents_to_disk=False,
            caption_extension="",
            enable_bucket=True,
            gradient_checkpointing=False,
            full_fp16=False,
            no_token_padding=False,
            stop_text_encoder_training_pct=0,
            xformers=True,
            save_model_as="safetensors",
            shuffle_caption=False,
            save_state=False,
            resume="",
            prior_loss_weight=1.0,
            text_encoder_lr=5e-05,
            unet_lr=0.0001,
            network_dim=128,
            lora_network_weights="",
            dim_from_weights=False,
            color_aug=False,
            flip_aug=False,
            clip_skip=1,
            gradient_accumulation_steps=1.0,
            mem_eff_attn=False,
            output_name='test',
            model_list="custom",
            max_token_length=75,
            max_train_epochs="",
            max_data_loader_n_workers=0,
            network_alpha=1,
            training_comment="",
            keep_tokens=0,
            lr_scheduler_num_cycles="",
            lr_scheduler_power="",
            persistent_data_loader_workers=False,
            bucket_no_upscale=True,
            random_crop=False,
            bucket_reso_steps=64,
            caption_dropout_every_n_epochs=0.0,
            caption_dropout_rate=0,
            optimizer="AdamW8bit",
            optimizer_args="",
            noise_offset_type="Original",
            noise_offset=0,
            adaptive_noise_scale=0,
            multires_noise_iterations=0,
            multires_noise_discount=0,
            LoRA_type="Standard",
            factor=-1,
            use_cp=False,
            decompose_both=False,
            train_on_input=False,
            conv_dim=1,
            conv_alpha=1,
            sample_every_n_steps=0,
            sample_every_n_epochs=0,
            sample_sampler="euler_a",
            sample_prompts="",
            additional_parameters="",
            vae_batch_size=0,
            min_snr_gamma=0,
            down_lr_weight="",
            mid_lr_weight="",
            up_lr_weight="",
            block_lr_zero_threshold="",
            block_dims="",
            block_alphas="",
            conv_dims="",
            conv_alphas="",
            weighted_captions=False,
            unit=1,
            save_every_n_steps=0,
            save_last_n_steps=0,
            save_last_n_steps_state=0,
            use_wandb=False,
            wandb_api_key="",
            scale_v_pred_loss_like_noise_pred=False,
            scale_weight_norms=0,
            network_dropout=0,
            rank_dropout=0,
            module_dropout=0,
            sdxl_cache_text_encoder_outputs=False,
            sdxl_no_half_vae=True,
            min_timestep=0,
            max_timestep=1000,
        )
        print("3333")
        # Assuming "test1.safetensors" is generated in "train_data/model/"
        original_file_path = os.path.join(
            train_data_dir, "model", "test1.safetensors")

        # Initialize a session using your credentials
        session = boto3.Session(
            aws_access_key_id='AKIAQH5LSIAPXFCSO4PS',
            aws_secret_access_key='CMTs+GqAQNdm7paMPGUu+mePGFSqw+arylPqCVQD',
            region_name='eu-north-1'
        )

        # Initialize the S3 client
        s3 = session.client('s3')

        # Define the file to be uploaded and the bucket to upload to
        bucket_name = 'runpod-models'
        train_id = os.path.basename(train_data_dir)
        # Upload the file
        s3.upload_file(Filename=original_file_path,
                       Bucket=bucket_name, Key=f"{train_id}.safetensors")

        # Extract the train_id from the train_data_dir
        

        # Destination path in "stable-diffusion-webui/models/lora/{train_id}/"
        dest_dir = os.path.join(
            base_dir, "..", "stable-diffusion-webui", "models", "lora", train_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, "test1.safetensors")

        # Move the file
        shutil.move(original_file_path, dest_file_path)

        print("Train finished!")

    async def create_train(self, request: Request):
        base_dir = os.path.abspath(os.curdir)

        train_id = str(uuid.uuid4())

        data = await request.json()

        train_images = data.get('train_images', [])
        save_dir = os.path.join(base_dir, 'train_images', train_id)

        os.makedirs(save_dir, exist_ok=True)

        for i, img_base64 in enumerate(train_images):
            img_data = base64.b64decode(img_base64)
            img_path = os.path.join(save_dir, f'image_{i}.jpg')

            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)


        train_data_dir = os.path.join(base_dir, "train_data", train_id)
        reg_data_dir = os.path.join(train_data_dir, "..", "reg_data")
        max_resolution = "512,512"
        model_path = os.path.join(base_dir, "model.safetensors")
        instance_prompt = "ohwx"
        class_prompt = "man"

        self.train_model_bg(save_dir=save_dir, train_data_dir=train_data_dir, reg_data_dir=reg_data_dir, model_path=model_path,
                            instance_prompt=instance_prompt, class_prompt=class_prompt, max_resolution=max_resolution)

        return models.TrainResponse(title="LoRA training", description="Completed", info="Success", train_id=train_id)

    def delete_training(self, id: str):
        base_dir = os.path.abspath(os.curdir)

        # Remove training data directory
        train_data_dir = os.path.join(base_dir, "train_data", id)
        model_data_dir = os.path.join(
            base_dir, "..", "stable-diffusion-webui", "models", "lora", id)
        if os.path.exists(train_data_dir):
            shutil.rmtree(train_data_dir)

        if os.path.exists(model_data_dir):
            shutil.rmtree(model_data_dir)

        return models.TrainInfoResponse(title="Remove LoRA training", description="Deleted", info="Success")

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name,
                    port=port, timeout_keep_alive=0)
