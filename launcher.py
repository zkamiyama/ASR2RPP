from asr2rpp.gui_dcc import main

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test-lifecycle':
        from asr2rpp.lifecycle_smoke import exercise
        exercise(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '--self-test-default':
        from asr2rpp.gui_acceptance import exercise
        exercise(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(main())
