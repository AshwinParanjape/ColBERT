import os
import random

from colbert.utils.parser import Arguments

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries, load_topK_pids
from colbert.ranking.reranking import rerank
from colbert.ranking.batch_reranking import batch_rerank

from meticulous import Experiment


def main():
    random.seed(12345)

    parser = Arguments(description='Re-ranking over a ColBERT index')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_reranking_input()
    parser.add_index_use_input()

    parser.add_argument('--step', dest='step', default=1, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--log-scores', dest='log_scores', default=False, action='store_true')
    parser.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    Experiment.add_argument_group(parser)
    args = parser.parse()
    args.depth = args.depth if args.depth > 0 else None
    vargs = {k:v for k, v in vars(args).items()}
    meticulous_args = {}
    for arg in ['project_directory', 'experiments_directory', 'experiment_id', 'description', 'resume', 'norecord']:
        if arg in vargs:
            meticulous_args[arg] = vargs[arg]
            del vargs[arg]
    experiment = Experiment(args=vargs, **meticulous_args)

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    args.colbert, args.checkpoint = load_colbert(args)

    args.qrels = load_qrels(args.qrels)
    args.queries = load_queries(args.queries)


    args.index_path = os.path.join(args.index_root, args.index_name)
    args.output_path = experiment.curexpdir

    if args.batch:
        batch_rerank(args)
    else:
        rerank(args)


if __name__ == "__main__":
    main()
