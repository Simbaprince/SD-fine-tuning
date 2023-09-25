import gradio as gr
import os
import argparse
from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from textual_inversion_gui import ti_tab
from library.utilities import utilities_tab
from lora_gui import lora_tab
from library.class_lora_tab import LoRATools

import os
from library.custom_logging import setup_logging
# from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, queue_lock  # noqa: F401
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
# from modules.shared import cmd_opts
from modules import cmd_args
import threading
from typing import Iterable
# Set up logging
parser = cmd_args.parser
cmd_opts, _ = parser.parse_known_args()

log = setup_logging()

queue_lock = threading.Lock()



def get_gradio_auth_creds() -> Iterable[tuple[str, ...]]:
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    def process_credential_line(s) -> tuple[str, ...] | None:
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    if cmd_opts.gradio_auth:
        for cred in cmd_opts.gradio_auth.split(','):
            cred = process_credential_line(cred)
            if cred:
                yield cred

    if cmd_opts.gradio_auth_path:
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                for cred in line.strip().split(','):
                    cred = process_credential_line(cred)
                    if cred:
                        yield cred

def configure_cors_middleware(app):
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)


def setup_middleware(app):
    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)

    app.build_middleware_stack()  # rebuild middleware stack on-the-fly
def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api
import time
def UI(**kwargs):
    while 1:

        css = ''

        headless = kwargs.get('headless', False)
        log.info(f'headless: {headless}')

        if os.path.exists('./style.css'):
            with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
                log.info('Load CSS...')
                css += file.read() + '\n'

        if os.path.exists('./.release'):
            with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
                release = file.read()

        if os.path.exists('./README.md'):
            with open(os.path.join('./README.md'), 'r', encoding='utf8') as file:
                README = file.read()

        interface = gr.Blocks(
            css=css, title=f'Kohya_ss GUI {release}', theme=gr.themes.Default()
        )

        with interface:
            with gr.Tab('Dreambooth'):
                (
                    train_data_dir_input,
                    reg_data_dir_input,
                    output_dir_input,
                    logging_dir_input,
                ) = dreambooth_tab(headless=headless)
            with gr.Tab('LoRA'):
                lora_tab(headless=headless)
            with gr.Tab('Textual Inversion'):
                ti_tab(headless=headless)
            with gr.Tab('Finetuning'):
                finetune_tab(headless=headless)
            with gr.Tab('Utilities'):
                utilities_tab(
                    train_data_dir_input=train_data_dir_input,
                    reg_data_dir_input=reg_data_dir_input,
                    output_dir_input=output_dir_input,
                    logging_dir_input=logging_dir_input,
                    enable_copy_info_button=True,
                    headless=headless,
                )
                with gr.Tab('LoRA'):
                    _ = LoRATools(headless=headless)
            with gr.Tab('About'):
                gr.Markdown(f'kohya_ss GUI release {release}')
                with gr.Tab('README'):
                    gr.Markdown(README)

            htmlStr = f"""
            <html>
                <body>
                    <div class="ver-class">{release}</div>
                </body>
            </html>
            """
            gr.HTML(htmlStr)
        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name
        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share
        import modules
        gradio_auth_creds = list(get_gradio_auth_creds()) or None
        app, local_url, share_url = interface.launch(share=cmd_opts.share,
                server_name=server_name,
                server_port=cmd_opts.port,
                ssl_keyfile=cmd_opts.tls_keyfile,
                ssl_certfile=cmd_opts.tls_certfile,
                ssl_verify=cmd_opts.disable_tls_verify,
                debug=cmd_opts.gradio_debug,
                auth=gradio_auth_creds,
                inbrowser=cmd_opts.autolaunch and os.getenv('SD_WEBUI_RESTARTING') != '1',
                prevent_thread_lock=True,
                allowed_paths=cmd_opts.gradio_allowed_path,
                app_kwargs={
                    "docs_url": "/docs",
                    "redoc_url": "/redoc",
                },
                root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
            )
        cmd_opts.autolaunch = False
        setup_middleware(app)    
        create_api(app)
        time.sleep(100000000)
if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '--headless', action='store_true', help='Is the server headless'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
