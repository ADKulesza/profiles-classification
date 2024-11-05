import numpy as np


def check_data(prof_arr, df, logger):
    if prof_arr.shape[0] == df.shape[0]:
        logger.info("Data sizes are matching! :)")
    else:
        raise ValueError("Mismatch data shape!", prof_arr.shape[0], df.shape[0])

    if not isinstance(prof_arr, (np.ndarray, np.generic)):
        raise ValueError(f"Wrong data type! {type(prof_arr)}")
