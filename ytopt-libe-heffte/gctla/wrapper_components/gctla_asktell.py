"""
This module wraps around the ytopt generator.
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
import logging
logger = logging.getLogger(__name__)

__all__ = ['persistent_gctla']

def persistent_gctla(H, persis_info, gen_specs, libE_info):
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user_specs = gen_specs['user']
    ensemble_dir = user_specs['ensemble_dir']
    n_sim = user_specs['num_sim_workers']
    model = user_specs['model']
    samples, sample_history = [], []
    sample_generation_identifier = 0
    tag = None
    calc_in = None
    first_write = True

    # Send batches until manager sends stop tag
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Hand off information
        H_o = np.zeros(n_sim, dtype=gen_specs['out'])
        filled = 0
        while filled < n_sim:
            # Replenish samples from the model as needed
            while len(samples) == 0:
                samples = model.sample_from_conditions(user_specs['conditions'])
                if 'remove_duplicates' in user_specs and user_specs['remove_duplicates'] is not None:
                    samples, sample_history = user_specs['remove_duplicates'](samples, sample_history, gen_specs['out'])
                # Leave an artifact of the sampling process
                samples.to_csv(f"GC_TLA_Samples_{sample_generation_identifier}.csv", index=False)
                sample_generation_identifier += 1
            # Use available samples
            utilized = []
            for idx in samples.index[:n_sim]:
                utilized.append(idx)
                for (key, value) in samples.loc[idx].items():
                    try:
                        H_o[filled][key] = value
                    except ValueError:
                        # mpi_ranks (would be KeyError but ndarrays raise ValueError instead)
                        pass
                filled += 1
                if filled >= n_sim:
                    break
            samples = samples.drop(index=utilized)
        print(f"[libE - generator {libE_info['workerID']}] creates points: {H_o}")
        # This returns the requested points to the libE manager, which will
        # perform the sim_f evaluations and then give back the values.
        tag, Work, calc_in = ps.send_recv(H_o)
        #print('received:', calc_in, flush=True)

        if calc_in is not None:
            if len(calc_in):
                b = []
                with open(f"persistent_H.npz", "wb") as npf:
                    np.save(npf, calc_in)
                for field_name, entry in zip(gen_specs['persis_in'], calc_in[0]):
                    try:
                        b += [str(entry[0])]
                    except Exception as e:
                        from inspect import currentframe
                        logger.warning(f"Field '{field_name}' with value '{entry}' produced exception {e.__class__} during persistent output in {__file__}:{currentframe().f_back.f_lineno}")
                        b += [str(entry)]
                # Drop in ensemble directory
                if first_write:
                    with open('../results.csv', 'w') as f:
                        f.write(",".join(calc_in.dtype.names)+ "\n")
                        first_write = False
                with open('../results.csv', 'a') as f:
                    f.write(",".join(b)+"\n")
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG

