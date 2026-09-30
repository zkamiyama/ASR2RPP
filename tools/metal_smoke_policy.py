"""Test selection from real Metal capabilities, never from a CI device name.

The pinned audio.cpp ggml-metal-device.m defines has_simdgroup_mm using
supportsFamily:MTLGPUFamilyApple7. Its Qwen encoder directly executes MUL_MAT
on Metal; unlike Whisper's scheduler, it cannot fall back for that operation.
"""

PROBE_SOURCE = r'''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
int main() { @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSDictionary *result = @{
        @"available": @(device != nil),
        @"device": device ? device.name : @"",
        @"simdgroup_matrix": @(device && [device supportsFamily:MTLGPUFamilyApple7])
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:result options:0 error:nil];
    if (!json) return 1;
    fwrite(json.bytes, 1, json.length, stdout);
    return 0;
}}
'''


def select_metal_cases(capabilities):
    """Return runnable and explicitly skipped models; malformed probes fail."""
    if (not isinstance(capabilities, dict)
            or type(capabilities.get('available')) is not bool
            or type(capabilities.get('simdgroup_matrix')) is not bool
            or not isinstance(capabilities.get('device'), str)):
        raise ValueError('Invalid Metal hardware capabilities')
    if not capabilities['available']:
        return [], {name: 'No Metal device is exposed to this runner'
                    for name in ('whisper-base', 'qwen3-asr-06b')}
    if not capabilities['simdgroup_matrix']:
        return ['whisper-base'], {'qwen3-asr-06b':
            'Device lacks the Apple7 SIMD-group matrix support required by the pinned Qwen Metal encoder'}
    return ['whisper-base', 'qwen3-asr-06b'], {}
