from src.utils import create_generators


def main():
    _, val_gen = create_generators()

    # Placeholder evaluasi: statistik batch validation
    total = val_gen.samples
    classes = val_gen.class_indices
    print("Total validation samples:", total)
    print("Classes:", classes)


if __name__ == '__main__':
    main()
