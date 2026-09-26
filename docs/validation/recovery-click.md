# Click-driven worker-start failure regression

A direct call to `MainWindow.start_work()` is not enough to test Qt lifecycle
recovery. When GO is actually clicked, `self.sender()` remains the button during
synchronous nested calls. A recovery path routed through the worker-signal slot
was therefore rejected by its stale-signal guard, leaving the item as waiting
instead of recording its failure and completing the attempt's counter.

The regression was reproduced on the pre-fix source by replacing the test's
method calls with `run_button.click()`: the expected failed item stayed waiting.
After separating `_apply_item_event` (internal state settlement) from `item_changed`
(the guarded incoming worker-signal slot), the same click-driven test passed,
including a successful second GO and exactly one terminal counter update.
The source's stale-worker guard was retained rather than removed.

The temporary reproduction job ran the failing pre-fix test, the successful
post-fix test, and the entire 210-test suite before committing the fix. The
one-time workflow is not part of the released tree. Windows frozen and native
release gates run separately on the final source commit.

This does not establish that every possible Qt or OS failure has been eliminated.
It documents a specifically reproduced inconsistent transition and its regression.
