"""Host backend policy shared by settings, CLI, registry and workers."""
import sys


def backends(platform=None):
    platform = sys.platform if platform is None else platform
    if platform == 'win32':
        return ('auto', 'cpu', 'vulkan')
    if platform == 'darwin':
        return ('auto', 'cpu', 'metal')
    return ('auto', 'cpu', 'vulkan', 'cuda')


def preferred_gpu(platform=None):
    return 'metal' if (platform or sys.platform) == 'darwin' else 'vulkan'


def validate_backend(device):
    if device not in backends():
        raise ValueError(f'Unsupported backend {device!r} on {sys.platform}; choose {", ".join(backends())}')


def automatic_devices(platform=None):
    platform = sys.platform if platform is None else platform
    if platform == 'win32':
        return ('vulkan', 'cpu')
    if platform == 'darwin':
        return ('metal', 'cpu')
    return ('cuda', 'vulkan', 'cpu')
