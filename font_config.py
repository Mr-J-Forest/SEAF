"""字体配置工具模块，为项目提供统一的中文字体设置入口。"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import matplotlib

logger = logging.getLogger(__name__)

# 默认使用无界面的 Agg 后端，适合服务器环境
try:
    matplotlib.use("Agg")
except Exception:  # pragma: no cover - 后端可能已初始化
    pass

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 常见的中文字体候选列表（按优先级排序）
_CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "Hiragino Sans GB",
    "PingFang SC",
    "Arial Unicode MS",
    "STHeiti",
    "STSong",
    "Liberation Sans",
    "FreeSans",
    "DejaVu Sans",
]


@functools.lru_cache(maxsize=1)
def setup_chinese_fonts(preferred_fonts: tuple[str, ...] | None = None) -> Optional[str]:
    """配置 Matplotlib 的中文字体支持。

    Args:
        preferred_fonts: 额外的字体优先列表，优先级高于默认候选列表。

    Returns:
        成功设置的字体名称，如果未找到中文字体则返回 ``None``。
    """

    candidates: list[str] = []
    if preferred_fonts:
        candidates.extend([font for font in preferred_fonts if font])
    candidates.extend(_CHINESE_FONT_CANDIDATES)

    available_fonts = {font.name for font in fm.fontManager.ttflist}
    logger.debug("Detected %d fonts from system", len(available_fonts))

    chosen_font: Optional[str] = None
    for font_name in candidates:
        if font_name in available_fonts:
            chosen_font = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"

    if chosen_font:
        plt.rcParams["font.sans-serif"] = [chosen_font] + candidates
        logger.info("中文字体已设置为 %s", chosen_font)
    else:
        plt.rcParams["font.sans-serif"] = candidates
        logger.warning("未找到中文字体，可能无法正常显示中文")

    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

    return chosen_font
