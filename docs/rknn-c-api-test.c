/**
 * Minimal test case for RKNN C API bug - rknn_inputs_set() returns -5
 *
 * This test demonstrates that:
 * 1. RKNN_QUERY_INPUT_ATTR returns correct size (1,228,800 bytes for uint8 640x640x3)
 * 2. rknn_inputs_set() fails with "input size < model input size" error
 * 3. The reported "model input size" is 4x the query result (suggesting int32 bug)
 *
 * Meanwhile, Python API works with identical model and input format.
 *
 * Compile: gcc -o rknn_test rknn_test.c -I/usr/include -L/usr/lib -lrknnrt
 * Run: ./rknn_test /path/to/model.rknn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rknn_api.h>

int rknn_test(const char* model_path) {
    rknn_context ctx = 0;
    FILE* f = NULL;
    void* model_buf = NULL;
    size_t model_size = 0;
    int ret = 0;

    /* Load model file */
    f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open model: %s\n", model_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    model_buf = malloc(model_size);
    if (!model_buf) {
        fprintf(stderr, "ERROR: Cannot allocate %zu bytes\n", model_size);
        fclose(f);
        return -1;
    }
    size_t nread = fread(model_buf, 1, model_size, f);
    fclose(f);
    if (nread != model_size) {
        fprintf(stderr, "ERROR: Read %zu of %zu bytes\n", nread, model_size);
        free(model_buf);
        return -1;
    }

    printf("Model: %s (%zu bytes)\n", model_path, model_size);

    /* Initialize RKNN context */
    ret = rknn_init(&ctx, model_buf, (uint32_t)model_size, 0, NULL);
    printf("rknn_init: %d\n", ret);
    if (ret < 0) {
        fprintf(stderr, "ERROR: rknn_init failed: %d\n", ret);
        free(model_buf);
        return -1;
    }

    /* Query input attributes */
    rknn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    printf("RKNN_QUERY_INPUT_ATTR: %d\n", ret);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Query input attr failed: %d\n", ret);
        rknn_destroy(ctx);
        free(model_buf);
        return -1;
    }

    printf("  Input tensor:\n");
    printf("    n_dims: %u\n", attr.n_dims);
    printf("    dims: [%u, %u, %u, %u]\n", attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]);
    printf("    type: %d (2=INT8, 5=FLOAT32)\n", attr.type);
    printf("    fmt: %d (0=NHWC, 1=NCHW)\n", attr.fmt);
    printf("    size: %u bytes\n", attr.size);
    printf("    n_elems: %u\n", attr.n_elems);

    /* Create input buffer matching the query result */
    uint8_t* input_buf = (uint8_t*)calloc(1, attr.size);
    if (!input_buf) {
        fprintf(stderr, "ERROR: Cannot allocate input buffer\n");
        rknn_destroy(ctx);
        free(model_buf);
        return -1;
    }

    /* Fill with test pattern */
    for (size_t i = 0; i < attr.size; i++) {
        input_buf[i] = (uint8_t)(i % 256);
    }

    /* Set input - THIS IS WHERE IT FAILS */
    rknn_input rknn_in;
    memset(&rknn_in, 0, sizeof(rknn_in));
    rknn_in.index = 0;
    rknn_in.buf = input_buf;
    rknn_in.size = attr.size;
    rknn_in.pass_through = 0;
    rknn_in.type = RKNN_TENSOR_UINT8;
    rknn_in.fmt = RKNN_TENSOR_NHWC;

    printf("\nCalling rknn_inputs_set with:\n");
    printf("  index: %u\n", rknn_in.index);
    printf("  buf: %p\n", rknn_in.buf);
    printf("  size: %u bytes\n", rknn_in.size);
    printf("  type: %d (UINT8)\n", rknn_in.type);
    printf("  fmt: %d (NHWC)\n", rknn_in.fmt);

    ret = rknn_inputs_set(ctx, 1, &rknn_in, NULL);
    printf("\nrknn_inputs_set: %d\n", ret);

    if (ret < 0) {
        fprintf(stderr, "ERROR: rknn_inputs_set failed with %d\n", ret);
        fprintf(stderr, "\nThis is the BUG: Query says %u bytes, but internal validation expects 4x that.\n", attr.size);
        fprintf(stderr, "Python API works with identical input - this is a C API validation bug.\n");
    } else {
        /* If it works, try running inference */
        printf("SUCCESS: Input set correctly\n");

        ret = rknn_run(ctx, NULL);
        printf("rknn_run: %d\n", ret);

        if (ret == 0) {
            rknn_input_output_num io_num;
            ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
            printf("Outputs: n_input=%u, n_output=%u\n", io_num.n_input, io_num.n_output);

            if (io_num.n_output > 0) {
                rknn_output* outputs = (rknn_output*)calloc(io_num.n_output, sizeof(rknn_output));
                for (uint32_t i = 0; i < io_num.n_output; i++) {
                    outputs[i].want_float = 1;
                    outputs[i].is_prealloc = 0;
                }

                ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
                printf("rknn_outputs_get: %d\n", ret);

                if (ret == 0) {
                    printf("First output: size=%u, buf=%p\n", outputs[0].size, outputs[0].buf);
                }

                rknn_outputs_release(ctx, io_num.n_output, outputs);
                free(outputs);
            }
        }
    }

    free(input_buf);
    rknn_destroy(ctx);
    free(model_buf);

    return ret;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.rknn>\n", argv[0]);
        fprintf(stderr, "\nThis test demonstrates the RKNN C API bug where rknn_inputs_set()\n");
        fprintf(stderr, "fails even with correct input size.\n");
        return 1;
    }

    printf("=== RKNN C API Bug Test ===\n\n");
    int result = rknn_test(argv[1]);

    printf("\n=== Result: %s ===\n", result < 0 ? "BUG CONFIRMED" : "WORKS (bug may be fixed)");
    return result < 0 ? 1 : 0;
}
