import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from .main import main
    raise SystemExit(main())
