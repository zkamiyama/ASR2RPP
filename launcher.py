from asr2rpp.gui_dcc import main

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test-lifecycle':
        from asr2rpp.lifecycle_smoke import exercise
        exercise(sys.argv[2])
    else:
        raise SystemExit(main())
