"""Install FlashMTP in SGLang multiprocessing children when explicitly enabled."""

import os

if os.environ.get("SGLANG_FLASHMTP_ACTIVE") == "1":
    from specforge.sglang_flashmtp.bootstrap import install

    install()
