TARGET := skernel2
TARGET_ELF := ../../target/riscv64gc-unknown-none-elf/release/$(TARGET)

all: clean $(TARGET) FORCE

$(TARGET): $(TARGET_ELF)
	@rust-objcopy --binary-architecture=riscv64 --strip-all -O binary $< $@

$(TARGET_ELF):
	@cargo build -p $(TARGET) --target riscv64gc-unknown-none-elf --release

clean:
	@rm -rf ./$(TARGET)
	@cargo clean -p $(TARGET) --target riscv64gc-unknown-none-elf --release

FORCE:

.PHONY: FORCE
