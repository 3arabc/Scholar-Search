import logging
import sys
from datetime import datetime
import os

# Create log directory if it doesn't exist
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# Generate logging file path with current date
current_date = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD, e.g., 20250407
logging_file_path = os.path.join(log_dir, f"search_pipe_{current_date}.log")

# logging_file_path = os.path.join(log_dir, f"server_pipe_test.log")

# ── 全局修复：Windows 下 GBK 编码无法处理 UTF-8 字符（中文字符等）──
# 将 stdout/stderr 及日志文件的编码统一为 UTF-8，避免 UnicodeEncodeError
if sys.platform == "win32":
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# Configure handlers with explicit UTF-8 encoding
handlers = [
    logging.FileHandler(logging_file_path, encoding="utf-8"),
    logging.StreamHandler(sys.stdout),
]

# Set logging level (DEBUG overrides INFO)
level = logging.INFO
# level = logging.DEBUG

# Configure basic logging
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

# Create logger
logger = logging.getLogger(__name__)

# Example usage
logger.debug("This is a debug message")
logger.info("This is an info message")
