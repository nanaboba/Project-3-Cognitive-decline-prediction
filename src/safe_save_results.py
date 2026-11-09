"""Protect authoritative tuning results and keep changed reruns separate."""
from pathlib import Path
import os
import tempfile
import pandas as pd


def _rerun_dir(output_file):
    # outputs/tuning/file.csv -> outputs/experiments/tuning_reruns/
    folder = output_file.parent.parent / 'experiments' / 'tuning_reruns'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _same_csv(a, b):
    try:
        x, y = pd.read_csv(a), pd.read_csv(b)
        pd.testing.assert_frame_equal(x, y, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)
        return True
    except (AssertionError, ValueError, pd.errors.ParserError, UnicodeError):
        return False


def safe_save_results(results_df, output_file):
    """Save a DataFrame without overwriting existing results.

    Returns the path retained. Identical reruns do not create new files.
    Comparison is made after CSV serialization to avoid dtype roundtrip noise.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Create candidate outside the authoritative folder, then compare.
    reruns = _rerun_dir(output_file)
    fd, temp = tempfile.mkstemp(prefix='.candidate_', suffix='.csv', dir=reruns)
    os.close(fd)
    candidate = Path(temp)
    try:
        results_df.to_csv(candidate, index=False)
        if not output_file.exists():
            # Exclusive creation prevents overwriting if another process created it.
            with output_file.open('xb') as dest, candidate.open('rb') as src:
                dest.write(src.read())
            print(f'Created authoritative results: {output_file}')
            return output_file
        try:
            same = _same_csv(output_file, candidate)
        except (OSError, PermissionError) as exc:
            raise RuntimeError(f'Cannot read authoritative results: {output_file}') from exc
        if same:
            print(f'Reproduced existing results; original preserved: {output_file}')
            return output_file
        for previous in sorted(reruns.glob(f'{output_file.stem}_rerun_*{output_file.suffix}')):
            if _same_csv(previous, candidate):
                print(f'Different from original but matches prior rerun: {previous}')
                return previous
        for n in range(1, 10000):
            destination = reruns / f'{output_file.stem}_rerun_{n:03d}{output_file.suffix}'
            try:
                with destination.open('xb') as dest, candidate.open('rb') as src:
                    dest.write(src.read())
                print(f'Original preserved: {output_file}\nDifferent results saved: {destination}')
                return destination
            except FileExistsError:
                continue
        raise RuntimeError('Too many rerun files')
    finally:
        candidate.unlink(missing_ok=True)


def safe_save_figure(fig, output_file, **savefig_kwargs):
    """Never overwrite a published tuning figure; put changed versions in reruns."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    reruns = _rerun_dir(output_file)
    fd, temp = tempfile.mkstemp(prefix='.figure_', suffix=output_file.suffix, dir=reruns)
    os.close(fd)
    candidate = Path(temp)
    try:
        fig.savefig(candidate, **savefig_kwargs)
        payload = candidate.read_bytes()
        if not output_file.exists():
            with output_file.open('xb') as f:
                f.write(payload)
            print(f'Created tuning figure: {output_file}')
            return output_file
        if output_file.read_bytes() == payload:
            print(f'Tuning figure reproduced: {output_file}')
            return output_file
        for previous in sorted(reruns.glob(f'{output_file.stem}_rerun_*{output_file.suffix}')):
            if previous.read_bytes() == payload:
                print(f'Figure matches prior rerun: {previous}')
                return previous
        for n in range(1, 10000):
            destination = reruns / f'{output_file.stem}_rerun_{n:03d}{output_file.suffix}'
            try:
                with destination.open('xb') as f:
                    f.write(payload)
                print(f'Original figure preserved; rerun saved: {destination}')
                return destination
            except FileExistsError:
                continue
        raise RuntimeError('Too many figure reruns')
    finally:
        candidate.unlink(missing_ok=True)
