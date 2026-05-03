import logging
import numpy as np

from src import TqdmToLogger

logger = logging.getLogger(__name__)



def simulate_split(args, dataset):
    """Split data indices using labels.
    
    Args:
        args (argparser): arguments
        dataset (dataset): raw dataset instance to be split 
        
    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
    """
    # IID split (i.e., statistical homogeneity)
    if args.split_type == 'iid': 
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))
        
        # get adjusted indices
        split_indices = np.array_split(shuffled_indices, args.K)
        
        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    
    # non-IID split by sample unbalancedness
    if args.split_type == 'unbalanced': 
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))
        
        # split indices by number of clients
        split_indices = np.array_split(shuffled_indices, args.K)
            
        # randomly remove some proportion (1% ~ 5%) of data
        keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))
            
        # get adjusted indices
        split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
        
        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    
    # Non-IID split proposed in (McMahan et al., 2016); each client has samples from at least two different classes
    elif args.split_type == 'patho': 
        try:
            assert args.mincls >= 2
        except AssertionError as e:
            logger.exception("[SIMULATE] Each client should have samples from at least 2 distinct classes!")
            raise e
        
        # get indices by class labels
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))
            
        # divide shards
        num_shards_per_class = args.K * args.mincls // args.num_classes
        if num_shards_per_class < 1:
            err = f'[SIMULATE] Increase the number of minimum class (`args.mincls` > {args.mincls}) or the number of participating clients (`args.K` > {args.K})!'
            logger.exception(err)
            raise Exception(err)
        
        # split class indices again into groups, each having the designated number of shards
        split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
        
        # make hashmap to track remaining shards to be assigned per client
        class_shards_counts = dict(zip([i for i in range(args.num_classes)], [len(split_idx) for split_idx in split_indices]))

        # assign divided shards to clients
        assigned_shards = []
        for _ in TqdmToLogger(
            range(args.K), 
            logger=logger,
            desc='[SIMULATE] ...assigning to clients... '
            ):
            # update selection probability according to the count of remaining shards
            # i.e., do NOT sample from class having no remaining shards
            remaining = np.array(list(class_shards_counts.values()))
            selection_prob = np.where(remaining > 0, 1., 0.)
            prob_sum = selection_prob.sum()

            if prob_sum == 0:
                # all shards exhausted — re-split the largest existing shard for remaining clients
                largest_idx = max(range(len(assigned_shards)), key=lambda i: len(assigned_shards[i]))
                half = len(assigned_shards[largest_idx]) // 2
                new_shard = assigned_shards[largest_idx][half:]
                assigned_shards[largest_idx] = assigned_shards[largest_idx][:half]
                assigned_shards.append(new_shard)
                continue

            selection_prob /= prob_sum

            # select classes to be considered
            try:
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=False, p=selection_prob)
            except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=True, p=selection_prob)

            # assign shards in randomly selected classes to current client
            for it, class_idx in enumerate(selected_classes):
                selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
                selected_shards = split_indices[class_idx].pop(selected_shard_indices)
                if it == 0:
                    assigned_shards.append([selected_shards])
                else:
                    assigned_shards[-1].append(selected_shards)
                class_shards_counts[class_idx] -= 1
            else:
                assigned_shards[-1] = np.concatenate(assigned_shards[-1])

        # construct a hashmap
        split_map = {k: assigned_shards[k] for k in range(args.K)}
        return split_map
    
    # Non-IID split proposed in (Hsu et al., 2019); simulation of non-IID split scenario using Dirichlet distribution
    elif args.split_type == 'diri':
        MIN_SAMPLES = int(1 / args.test_size) if args.test_size > 0 else 0

        # get indices by class labels
        total_counts = len(dataset.targets)
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)

        # calculate ideal samples counts per client
        ideal_counts = len(dataset.targets) // args.K
        if ideal_counts < 1:
            err = f'[SIMULATE] Decrease the number of participating clients (`args.K` < {args.K})!'
            logger.exception(err)
            raise Exception(err)

        # `satisfied_counts` tracks requested (not obtained) samples, so late
        # clients can still end up with 0 once class pools drain. Retry the
        # whole split with a fresh RNG draw until every client is non-empty.
        MAX_ATTEMPTS = 100
        for attempt in range(MAX_ATTEMPTS):
            class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))
            assigned_indices = []

            ## NOTE: it is possible that not all samples be consumed, as it is intended for satisfying each clients having at least `MIN_SAMPLES` samples per class
            for k in TqdmToLogger(range(args.K), logger=logger, desc='[SIMULATE] ...assigning to clients... '):
                ### for current client of which index is `k`
                curr_indices = []
                satisfied_counts = 0

                ### ...until the number of samples close to ideal counts is filled
                while satisfied_counts < ideal_counts:
                    ### define Dirichlet distribution of which prior distribution is an uniform distribution
                    diri_prior = np.random.uniform(size=args.num_classes)

                    ### sample a parameter corresponded to that of categorical distribution
                    cat_param = np.random.dirichlet(alpha=args.cncntrtn * diri_prior)

                    ### try to sample by amount of `ideal_counts``
                    sampled = np.random.choice(args.num_classes, ideal_counts, p=cat_param)

                    ### count per-class samples
                    unique, counts = np.unique(sampled, return_counts=True)
                    if len(unique) < args.mincls:
                        continue

                    ### filter out sampled classes not having as much as `MIN_SAMPLES`
                    required_counts = counts * (counts > MIN_SAMPLES)

                    ### assign from population indices split by classes
                    for idx, required_class in enumerate(unique):
                        if required_counts[idx] == 0: continue
                        sampled_indices = class_indices[required_class][:required_counts[idx]]
                        curr_indices.append(sampled_indices)
                        class_indices[required_class] = class_indices[required_class][required_counts[idx]:]
                    satisfied_counts += sum(required_counts)

                ### when enough samples are collected, go to next clients!
                assigned_indices.append(
                    np.concatenate(curr_indices) if curr_indices else np.empty(0, dtype=np.int64)
                )

            empties = sum(1 for a in assigned_indices if len(a) == 0)
            if empties == 0:
                if attempt > 0:
                    logger.info(f'[SIMULATE] diri split succeeded after {attempt + 1} attempts')
                break
            logger.warning(f'[SIMULATE] diri attempt {attempt + 1}: {empties} empty client(s), retrying')
        else:
            raise RuntimeError(
                f'[SIMULATE] diri split failed to produce all-non-empty clients after {MAX_ATTEMPTS} attempts'
            )

        # construct a hashmap
        split_map = {k: assigned_indices[k] for k in range(args.K)}
        return split_map
    # Dataset-specific real-world heterogeneity
    elif args.split_type == 'custom':
        return _custom_split(args, dataset)

    # `leaf` - LEAF benchmark (Caldas et al., 2018); `fedvis` - Federated Vision Datasets (Hsu, Qi and Brown, 2020)
    elif args.split_type in ['leaf']:
        logger.info('[SIMULATE] Use pre-defined split!')


def _custom_split(args, dataset):
    """Dispatch to the dataset-specific `custom` heterogeneous split.

    MIMIC4   : Strict care-unit allocation — every client is assigned the
               samples of exactly one care unit.  Clients per care unit are
               allocated proportional to the unit's size (floor 1).  This
               maximises the per-client drift-severity gradient because the
               ICD-9 -> ICD-10 corrupt-group prevalence is strongly
               correlated with care unit (e.g. TSICU ~= 100% corrupt,
               CCU/CVICU ~= 0%).  `--cncntrtn` is unused.
    CheXpert : Dirichlet over `has_lateral` (each client's mix of samples
               with vs. without a lateral twin is drawn from Dir(alpha);
               combined with the lateral-swap drift, this yields a per-client
               drift-severity gradient).
    """
    if args.dataset == 'MIMIC4':
        return _mimic_strict_by_careunit(args, dataset)
    if args.dataset == 'CheXpert':
        return _chexpert_dirichlet_by_pairing(args, dataset)
    raise NotImplementedError(
        f'[SIMULATE] `custom` split is not implemented for dataset `{args.dataset}`.'
    )


def _mimic_strict_by_careunit(args, dataset):
    """Strict one-care-unit-per-client partition of ICU admissions.

    Every client is assigned samples from exactly one `first_careunit`.
    Care units are allocated multiple clients in proportion to their sample
    count (largest-remainder method, floor 1).  Within a care unit the
    samples are shuffled and split evenly across the clients allocated to
    that unit.

    This maximises the per-client drift-severity gradient: corrupt-group
    prevalence is strongly correlated with care unit (TSICU ~= all corrupt,
    CCU/CVICU ~= none), so strict partitioning produces clients that are
    effectively immune or fully exposed to the ICD-9 -> ICD-10 drift.

    `--cncntrtn` is not used by this split.
    """
    care_units = getattr(dataset, 'care_units', None)
    if care_units is None:
        raise AttributeError(
            '[SIMULATE] MIMIC4 dataset is missing `care_units`; '
            'cannot build custom care-unit split.'
        )
    care_units = np.asarray(care_units)
    unique_cus, inverse, counts = np.unique(
        care_units, return_inverse=True, return_counts=True
    )
    K = int(args.K)
    if K < len(unique_cus):
        raise ValueError(
            f'[SIMULATE] [MIMIC4] strict care-unit split requires K >= '
            f'{len(unique_cus)} (one client per care unit); got K={K}.'
        )
    rng = np.random.default_rng(args.seed)

    # Largest-remainder allocation: floor 1 per unit, distribute the rest
    # proportional to unit size.
    alloc = np.ones(len(unique_cus), dtype=int)
    remaining = K - len(unique_cus)
    if remaining > 0:
        quotas = counts.astype(float) / counts.sum() * remaining
        floors = np.floor(quotas).astype(int)
        alloc += floors
        leftover = remaining - floors.sum()
        if leftover > 0:
            top = np.argsort(-(quotas - floors))[:leftover]
            alloc[top] += 1

    assigned = [None] * K
    next_k = 0
    for c_idx, (cu, n_clients) in enumerate(zip(unique_cus, alloc)):
        idx_c = np.where(inverse == c_idx)[0]
        rng.shuffle(idx_c)
        chunks = np.array_split(idx_c, int(n_clients))
        for chunk in chunks:
            assigned[next_k] = chunk.astype(np.int64)
            next_k += 1

    logger.info(
        '[SIMULATE] [MIMIC4] custom split: strict care-unit allocation '
        'across %d clients.', K,
    )
    for c_idx, cu in enumerate(unique_cus):
        logger.info(
            '[SIMULATE] [MIMIC4]   %s: n=%d samples -> %d clients',
            cu, int(counts[c_idx]), int(alloc[c_idx]),
        )
    for k in range(K):
        cu_label = unique_cus[inverse[assigned[k][0]]] if len(assigned[k]) else '<empty>'
        logger.debug(
            '[SIMULATE] [MIMIC4] client %03d: n=%d | %s',
            k, len(assigned[k]), cu_label,
        )

    return {k: assigned[k] for k in range(K)}


def _chexpert_dirichlet_by_pairing(args, dataset):
    """Dirichlet(alpha) partition of CheXpert frontal samples by lateral-twin.

    The binary axis `has_lateral` (True iff the sample's study also contains
    a lateral view) is the sole non-IID dimension.  For each group g in
    {paired, unpaired} we sample q_g ~ Dir(alpha * 1_K) and distribute that
    group's records across the K clients proportional to q_g.  Small alpha
    yields strongly skewed mixes (some clients nearly all-paired, others
    nearly all-unpaired); large alpha approaches uniform.  Because lateral
    pairing is the drift predicate, the resulting paired-fraction gradient
    across clients is also the effective drift-severity gradient after
    `drift_start`.
    """
    has_lateral = getattr(dataset, 'has_lateral', None)
    if has_lateral is None:
        raise AttributeError(
            '[SIMULATE] CheXpert dataset is missing `has_lateral`; '
            'cannot build custom pairing-based split.'
        )
    has_lateral = np.asarray(has_lateral, dtype=bool)
    groups = has_lateral.astype(np.int64)  # 0 = unpaired, 1 = paired
    K, alpha = int(args.K), float(args.cncntrtn)
    rng = np.random.default_rng(args.seed)

    assigned = [[] for _ in range(K)]
    for g_val, g_name in [(0, 'unpaired'), (1, 'paired')]:
        idx_g = np.where(groups == g_val)[0]
        if len(idx_g) == 0:
            continue
        rng.shuffle(idx_g)
        q = rng.dirichlet([alpha] * K)
        cuts = np.round(np.cumsum(q) * len(idx_g)).astype(int)
        cuts[-1] = len(idx_g)
        prev = 0
        for k in range(K):
            assigned[k].extend(idx_g[prev:cuts[k]].tolist())
            prev = cuts[k]

    # Drop empty clients by redistributing a few samples from the largest.
    sizes = [len(a) for a in assigned]
    for k in range(K):
        if sizes[k] == 0:
            donor = int(np.argmax(sizes))
            share = max(1, sizes[donor] // 50)
            assigned[k].extend(assigned[donor][-share:])
            del assigned[donor][-share:]
            sizes = [len(a) for a in assigned]

    total_paired = int(has_lateral.sum())
    total = len(has_lateral)
    logger.info(
        '[SIMULATE] [CHEXPERT] custom split: Dir(alpha=%.3f) over {unpaired, paired} '
        '(global paired share %.1f%% = %d/%d) across %d clients.',
        alpha, 100.0 * total_paired / max(1, total), total_paired, total, K,
    )
    for k in range(K):
        n = len(assigned[k])
        n_paired = int(has_lateral[np.asarray(assigned[k], dtype=np.int64)].sum()) if n else 0
        logger.debug(
            '[SIMULATE] [CHEXPERT] client %03d: n=%d | paired=%d (%.1f%%)',
            k, n, n_paired, 100.0 * n_paired / max(1, n),
        )

    return {k: np.asarray(assigned[k], dtype=np.int64) for k in range(K)}
