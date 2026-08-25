#include <stdio.h>
#include <inttypes.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static void mem_cb(qemu_plugin_id_t id, struct qemu_plugin_meminfo *info,
                   uint64_t vaddr, void *userdata)
{
    uint64_t pc = qemu_plugin_get_insn_vaddr(info);
    int is_store = qemu_plugin_mem_is_store(info);

    printf("pc=0x%" PRIx64 " addr=0x%" PRIx64 " %s\n",
           pc, vaddr, is_store ? "W" : "R");
}

static void insn_exec_cb(qemu_plugin_id_t id,
                         struct qemu_plugin_insn *insn,
                         void *userdata)
{
    qemu_plugin_register_vcpu_mem_cb(
        insn, mem_cb, QEMU_PLUGIN_CB_NO_REGS, NULL);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    qemu_plugin_register_vcpu_insn_exec_cb(
        id, insn_exec_cb, QEMU_PLUGIN_CB_NO_REGS, NULL);
    return 0;
}
