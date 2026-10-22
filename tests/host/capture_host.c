// Minimal CLAP host that exercises the Flux AI Bridge capture flow end to end:
// load .clap -> activate -> set Capture on -> feed audio -> set Capture off ->
// wait for the background worker to write output files.
//
// Usage: capture_host <plugin.clap> <out_dir_to_watch> [seconds_of_audio]
// The plugin reads FLUX_AI_BIN / FLUX_AI_OUT from the environment.

#include <clap/clap.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>

#define SR 44100.0
#define BLOCK 512

static const void *host_get_extension(const clap_host_t *h, const char *id) { (void)h; (void)id; return NULL; }
static void host_noop(const clap_host_t *h) { (void)h; }

static const clap_host_t HOST = {
    .clap_version = CLAP_VERSION_INIT, .host_data = NULL, .name = "flux-test-host", .vendor = "flux",
    .url = "", .version = "0", .get_extension = host_get_extension,
    .request_restart = host_noop, .request_process = host_noop, .request_callback = host_noop,
};

typedef struct { clap_input_events_t base; clap_event_param_value_t ev; int pending; } in_events_t;
static uint32_t in_size(const clap_input_events_t *l) { return ((in_events_t *)l->ctx)->pending ? 1 : 0; }
static const clap_event_header_t *in_get(const clap_input_events_t *l, uint32_t i) {
    (void)i; return &((in_events_t *)l->ctx)->ev.header;
}
static bool out_try_push(const clap_output_events_t *l, const clap_event_header_t *h) { (void)l; (void)h; return true; }

static long count_wavs(const char *root);

static long count_wavs_dir(const char *dir) {
    long n = 0; DIR *d = opendir(dir); if (!d) return 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char p[4096]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        size_t l = strlen(e->d_name);
        if (e->d_type == DT_DIR) n += count_wavs_dir(p);
        else if (l > 4 && !strcmp(e->d_name + l - 4, ".wav") && strcmp(e->d_name, "capture.wav")) n++;
    }
    closedir(d); return n;
}
static long count_wavs(const char *root) { return count_wavs_dir(root); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s plugin.clap out_dir [secs]\n", argv[0]); return 2; }
    const char *path = argv[1], *outdir = argv[2];
    double secs = argc > 3 ? atof(argv[3]) : 8.0;

    // On macOS a .clap is a bundle; the binary lives in Contents/MacOS.
    char bin[4096]; snprintf(bin, sizeof bin, "%s/Contents/MacOS/%s", path, "Flux AI Bridge");
    void *lib = dlopen(bin, RTLD_NOW);
    if (!lib) { snprintf(bin, sizeof bin, "%s", path); lib = dlopen(bin, RTLD_NOW); }
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    const clap_plugin_entry_t *entry = dlsym(lib, "clap_entry");
    if (!entry || !entry->init(path)) { fprintf(stderr, "bad entry\n"); return 1; }
    const clap_plugin_factory_t *fac = entry->get_factory(CLAP_PLUGIN_FACTORY_ID);
    const clap_plugin_descriptor_t *desc = fac->get_plugin_descriptor(fac, 0);
    printf("plugin: %s (%s)\n", desc->name, desc->id);
    const clap_plugin_t *p = fac->create_plugin(fac, &HOST, desc->id);
    if (!p || !p->init(p)) { fprintf(stderr, "plugin init failed\n"); return 1; }

    const clap_plugin_params_t *params = p->get_extension(p, CLAP_EXT_PARAMS);
    clap_id capture_id = CLAP_INVALID_ID;
    for (uint32_t i = 0; params && i < params->count(p); i++) {
        clap_param_info_t info; params->get_info(p, i, &info);
        printf("param %u: %s\n", info.id, info.name);
        if (!strcmp(info.name, "Capture")) capture_id = info.id;
    }
    if (capture_id == CLAP_INVALID_ID) { fprintf(stderr, "no Capture param\n"); return 1; }

    if (!p->activate(p, SR, 1, BLOCK) || !p->start_processing(p)) { fprintf(stderr, "activate failed\n"); return 1; }

    float inl[BLOCK], inr[BLOCK], outl[BLOCK], outr[BLOCK];
    float *in_ch[2] = {inl, inr}, *out_ch[2] = {outl, outr};
    clap_audio_buffer_t in_buf = {.data32 = in_ch, .channel_count = 2}, out_buf = {.data32 = out_ch, .channel_count = 2};
    in_events_t ev = {0};
    ev.base.ctx = &ev; ev.base.size = in_size; ev.base.get = in_get;
    clap_output_events_t oev = {.ctx = NULL, .try_push = out_try_push};
    clap_process_t proc = {.frames_count = BLOCK, .audio_inputs = &in_buf, .audio_inputs_count = 1,
                           .audio_outputs = &out_buf, .audio_outputs_count = 1,
                           .in_events = &ev.base, .out_events = &oev};

    long frame = 0, total = (long)(secs * SR);
    double ph = 0;
    int stage = 0; // 0 = send capture on, 1 = feed audio, 2 = send capture off, 3 = wait
    long blocks_waited = 0;
    time_t t0 = time(NULL);
    for (;;) {
        // Test signal: 220 Hz tone + noise (same character as the python-side test clip).
        for (int i = 0; i < BLOCK; i++) {
            ph += 2 * M_PI * 220.0 / SR;
            float s = (float)(0.4 * sin(ph) + 0.02 * ((rand() / (double)RAND_MAX) - 0.5));
            inl[i] = inr[i] = s;
        }
        ev.pending = 0;
        if (stage == 0 || stage == 2) {
            ev.ev.header = (clap_event_header_t){.size = sizeof(ev.ev), .time = 0, .space_id = CLAP_CORE_EVENT_SPACE_ID,
                                                 .type = CLAP_EVENT_PARAM_VALUE, .flags = 0};
            ev.ev.param_id = capture_id; ev.ev.cookie = NULL; ev.ev.note_id = -1; ev.ev.port_index = -1;
            ev.ev.channel = -1; ev.ev.key = -1; ev.ev.value = stage == 0 ? 1.0 : 0.0;
            ev.pending = 1;
        }
        if (p->process(p, &proc) == CLAP_PROCESS_ERROR) { fprintf(stderr, "process error\n"); return 1; }
        if (stage == 0) stage = 1;
        else if (stage == 1) { frame += BLOCK; if (frame >= total) stage = 2; }
        else if (stage == 2) { stage = 3; printf("capture off after %.1fs of audio; waiting for worker...\n", frame / SR); }
        else {
            blocks_waited++;
            if (blocks_waited % 100 == 0) {
                long n = count_wavs(outdir);
                if (n >= 4) { printf("OK: %ld output wav files under %s\n", n, outdir); break; }
                if (difftime(time(NULL), t0) > 240) { fprintf(stderr, "FAIL: timed out, %ld files\n", n); return 1; }
            }
            usleep(10000); // ~real-time pacing while the worker runs
        }
    }
    p->stop_processing(p); p->deactivate(p); p->destroy(p); entry->deinit();
    return 0;
}
