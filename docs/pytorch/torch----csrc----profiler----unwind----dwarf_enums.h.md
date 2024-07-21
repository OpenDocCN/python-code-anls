# `.\pytorch\torch\csrc\profiler\unwind\dwarf_enums.h`

```py
#pragma once

// Exception Handling Pointer Encoding (DW_EH_PE) constants
enum {
  DW_EH_PE_absptr = 0x00,       // Absolute pointer
  DW_EH_PE_omit = 0xff,         // Value omitted
  DW_EH_PE_uleb128 = 0x01,      // Unsigned LEB128
  DW_EH_PE_udata2 = 0x02,       // Unsigned data (2 bytes)
  DW_EH_PE_udata4 = 0x03,       // Unsigned data (4 bytes)
  DW_EH_PE_udata8 = 0x04,       // Unsigned data (8 bytes)
  DW_EH_PE_sleb128 = 0x09,      // Signed LEB128
  DW_EH_PE_sdata2 = 0x0a,       // Signed data (2 bytes)
  DW_EH_PE_sdata4 = 0x0b,       // Signed data (4 bytes)
  DW_EH_PE_sdata8 = 0x0c,       // Signed data (8 bytes)
  DW_EH_PE_signed = 0x08,       // Signed
  DW_EH_PE_pcrel = 0x10,        // PC-relative
  DW_EH_PE_textrel = 0x20,      // Text-relative
  DW_EH_PE_datarel = 0x30,      // Data-relative
  DW_EH_PE_funcrel = 0x40,      // Function-relative
  DW_EH_PE_aligned = 0x50,      // Aligned
  DW_EH_PE_indirect = 0x80,     // Indirect
};

// Call Frame Instruction (DW_CFA) constants
enum {
  DW_CFA_nop = 0x0,             // No operation
  DW_CFA_advance_loc = 0x01,    // Advance location
  DW_CFA_offset = 0x02,         // Offset
  DW_CFA_restore = 0x03,        // Restore
  DW_CFA_advance_loc1 = 0x02,   // Advance location (1 byte)
  DW_CFA_advance_loc2 = 0x03,   // Advance location (2 bytes)
  DW_CFA_advance_loc4 = 0x04,   // Advance location (4 bytes)
  DW_CFA_restore_extended = 0x06, // Restore extended
  DW_CFA_undefined = 0x07,      // Undefined
  DW_CFA_register = 0x09,       // Register
  DW_CFA_remember_state = 0x0a, // Remember state
  DW_CFA_restore_state = 0x0b,  // Restore state
  DW_CFA_def_cfa = 0x0c,        // Define CFA
  DW_CFA_def_cfa_register = 0x0d, // Define CFA register
  DW_CFA_def_cfa_offset = 0x0e, // Define CFA offset
  DW_CFA_def_cfa_expression = 0xf, // Define CFA expression
  DW_CFA_expression = 0x10,     // Expression
  DW_CFA_offset_extended_sf = 0x11, // Offset extended (signed)
  DW_CFA_GNU_args_size = 0x2e,  // GNU args size
  DW_OP_deref = 0x6,            // Dereference
};
```