from __future__ import annotations

import argparse
import os
import socket
import traceback

from external_pp.remote.rpc import parse_endpoint, recv_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description="HXinfer remote vLLM worker service")
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--listen", required=True, help="control endpoint HOST:PORT")
    parser.add_argument("--visible-device", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device

    from vllm.v1.outputs import AsyncModelRunnerOutput
    from vllm.v1.serial_utils import run_method
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    wrapper = WorkerWrapperBase(rpc_rank=args.rank, global_rank=args.rank)
    initialized = False
    host, port = parse_endpoint(args.listen)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    print(f"HXINFER_WORKER_READY rank={args.rank} endpoint={args.listen}", flush=True)
    connection, peer = listener.accept()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"HXINFER_CONTROLLER_CONNECTED rank={args.rank} peer={peer}", flush=True)

    try:
        while True:
            request = recv_message(connection)
            request_id = request["id"]
            method = request["method"]
            method_name = method if isinstance(method, str) else "<callable>"
            print(
                f"HXINFER_RPC_BEGIN rank={args.rank} id={request_id} "
                f"method={method_name}",
                flush=True,
            )
            try:
                if method == "__ping__":
                    result = {"rank": args.rank, "initialized": initialized}
                elif method == "__shutdown_service__":
                    if initialized:
                        wrapper.shutdown()
                    send_message(
                        connection,
                        {"id": request_id, "ok": True, "result": None},
                    )
                    break
                else:
                    result = run_method(
                        wrapper,
                        method,
                        request.get("args", ()),
                        request.get("kwargs", {}),
                    )
                    if method == "init_worker":
                        initialized = True
                    if isinstance(result, AsyncModelRunnerOutput):
                        result = result.get_output()
                send_message(
                    connection,
                    {"id": request_id, "ok": True, "result": result},
                )
                print(
                    f"HXINFER_RPC_END rank={args.rank} id={request_id} "
                    f"method={method_name} ok=true",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                send_message(
                    connection,
                    {
                        "id": request_id,
                        "ok": False,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    f"HXINFER_RPC_END rank={args.rank} id={request_id} "
                    f"method={method_name} ok=false error={error!r}",
                    flush=True,
                )
    except (ConnectionError, EOFError):
        pass
    finally:
        if initialized:
            try:
                wrapper.shutdown()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        connection.close()
        listener.close()


if __name__ == "__main__":
    main()
