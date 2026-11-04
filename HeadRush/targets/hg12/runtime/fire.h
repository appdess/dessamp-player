#ifndef DESS_FIRE_HG12_H
#define DESS_FIRE_HG12_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One independent instance per mono amp block. Create/destroy outside the audio callback.
   The signed-off input contract for this private prototype is the exact packaged FIRE v1
   at 48000 Hz. Input/output are mono float32. The normalized vector order is
   Lead Gain, Bass, Middle, Treble, Presence. Keep the same instance through knob moves. */
typedef struct fire_instance fire_instance;
__attribute__((visibility("default"))) fire_instance *fire_create(const unsigned char *dess, size_t bytes, int max_block);
__attribute__((visibility("default"))) int fire_process(fire_instance *amp, float *input, float *output, int frames,
                                                       const float controls[5]);
__attribute__((visibility("default"))) void fire_destroy(fire_instance *amp);
#ifdef __cplusplus
}
#endif
#endif
