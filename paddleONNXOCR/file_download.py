import os
import re
import uuid
import aiohttp
import asyncio
from pathlib import Path
from urllib.parse import urlsplit, unquote
import aiofiles


class AsyncFileDownloader:
    def __init__(self, timeout=86400, chunk_size=8192, default_download_dir="downloads"):
        """
        :param timeout: 请求超时时间（秒）
        :param chunk_size: 每次读取的字节数（用于流式下载）
        :param default_download_dir: 默认下载目录（相对于当前工作目录）
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.chunk_size = chunk_size
        self.default_download_dir = Path(default_download_dir).resolve()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        }

        # MIME 类型到扩展名的映射（用于 fallback）
        self.mime_to_ext = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/svg+xml': '.svg',
            'video/mp4': '.mp4',
            'video/webm': '.webm',
            'video/ogg': '.ogg',
            'video/x-msvideo': '.avi',
            'video/mpeg': '.mpeg',
            'video/quicktime': '.mov',
            'video/x-matroska': '.mkv',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'application/zip': '.zip',
            'application/x-rar-compressed': '.rar',
            'application/octet-stream': '.bin',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            # 可继续扩展
        }

    async def download(self, url: str, save_path=None, overwrite: bool = False):
        """
        异步下载文件并保存到本地。

        :param url: 要下载的文件 URL
        :param save_path:
            - None → 使用默认目录 + 自动生成唯一文件名
            - 目录路径 → 保存到该目录 + 自动生成文件名
            - 完整文件路径 → 保存为指定名称
        :param overwrite: 是否覆盖已存在的文件
        :return: 实际保存的文件路径（str）
        """
        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {response.reason}")

                    # 获取原始建议文件名
                    raw_filename = self._get_filename_from_response(response, url)
                    _, ext = os.path.splitext(raw_filename)

                    # 如果没有有效扩展名，则尝试从 Content-Type 推断
                    if not ext or ext.lower() == '.':
                        content_type = response.headers.get('Content-Type', '')
                        mime_type = content_type.split(';')[0].strip().lower()
                        ext = self.mime_to_ext.get(mime_type, '.bin')

                    # 确定最终保存路径
                    if save_path is not None and not Path(save_path).is_dir():
                        final_path = Path(save_path)
                    else:
                        unique_name = f"{uuid.uuid4().hex}{ext}"
                        target_dir = Path(save_path) if save_path is not None else self.default_download_dir
                        target_dir.mkdir(parents=True, exist_ok=True)
                        final_path = target_dir / unique_name

                    # 检查是否已存在
                    if final_path.exists() and not overwrite:
                        raise FileExistsError(f"文件已存在且不覆盖: {final_path}")

                    # 创建父目录（确保存在）
                    final_path.parent.mkdir(parents=True, exist_ok=True)

                    # 使用 aiofiles 异步打开并写入
                    async with aiofiles.open(final_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            await f.write(chunk)
                        await f.flush()  # 可选：立即刷盘

                    return str(final_path)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"下载失败（网络错误）: {e}")
            except asyncio.TimeoutError:
                raise RuntimeError("下载超时")
            except Exception as e:
                raise RuntimeError(f"下载失败: {e}")

    def _get_filename_from_response(self, response, url):
        """从响应头或 URL 中提取并解码文件名"""
        content_disposition = response.headers.get('Content-Disposition', '')

        # 1. 处理 RFC 5987: filename*=UTF-8''...
        rfc5987_match = re.search(r"filename\*=UTF-8''([^\s;]+)", content_disposition, re.IGNORECASE)
        if rfc5987_match:
            encoded = rfc5987_match.group(1)
            try:
                return unquote(encoded)
            except Exception:
                pass

        # 2. 处理普通 filename="..."
        plain_match = re.search(r'filename\s*=\s*"?([^"\n;]+)"?', content_disposition, re.IGNORECASE)
        if plain_match:
            name = plain_match.group(1).strip()
            try:
                return unquote(name)
            except Exception:
                return name

        # 3. 从 URL 提取 basename 并解码
        parsed = urlsplit(url)
        basename = os.path.basename(parsed.path)
        if basename:
            try:
                decoded = unquote(basename)
                if decoded and any(c.isalnum() for c in decoded):  # 基本有效性判断
                    return decoded
            except Exception:
                pass

        # 4. 默认返回空（后续用 uuid + ext 替代）
        return ""


# 全局实例
file_downloader = AsyncFileDownloader()
