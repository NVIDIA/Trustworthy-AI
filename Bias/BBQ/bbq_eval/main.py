import argparse
import asyncio
import logging
import os


def main_bbq(
    categories,
    config_file,
    output_dir,
    dataset,
    num_rows,
    small_dataset,
    resume=False,
    checkpoint_interval=50,
    main_concurrency=20,
    parser_concurrency=40,
    window_size=100,
    use_categories=True,
):
    from bbq_eval.evaluate import main as evaluate_main
    from bbq_eval.openai.generate_responses_optimized import main_async as generate_main

    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), "openai/default_config.yaml")
    if small_dataset:
        dataset = os.path.join(
            os.path.dirname(__file__), "data", "dataset_small_ambig_95ci_10er.json"
        )
    elif dataset is None:
        dataset = "heegyu/bbq"
    if categories is None:
        categories = [
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Race_x_SES",
            "Race_x_gender",
            "Religion",
            "SES",
            "Sexual_orientation",
        ]
    logging.info(
        f"Starting BBQ generation for categories: {categories}, dataset: {dataset}, config: {config_file}"
    )
    logging.info(
        f"Pipeline config: main_concurrency={main_concurrency}, parser_concurrency={parser_concurrency}, checkpoint_interval={checkpoint_interval}"
    )

    # Run optimized pipeline
    output_file = asyncio.run(
        generate_main(
            categories=categories,
            config_file=config_file,
            output_dir=output_dir,
            dataset=dataset,
            num_rows=num_rows,
            resume=resume,
            main_concurrency=main_concurrency,
            parser_concurrency=parser_concurrency,
            checkpoint_interval=checkpoint_interval,
            window_size=window_size,
        )
    )

    # get bbq path
    bbq_path = os.path.dirname(__file__)
    metadata_path = os.path.join(bbq_path, "additional_data/additional_metadata.csv")
    logging.info(f"Starting evaluation with metadata: {metadata_path}, responses: {output_file}")
    logging.info(
        f"Evaluation mode: {'per-category (2 files)' if use_categories else 'overall-only (1 file)'}"
    )
    evaluate_main(
        metadata=metadata_path,
        input_file=output_file,
        output_dir=output_dir,
        name="model_response",
        use_categories=use_categories,
    )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Generate Responses")
    parser.add_argument("-d", "--dataset", type=str, default="heegyu/bbq", help="Dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config filename (defaults to bundled configuration)",
    )
    parser.add_argument(
        "-a",
        "--categories",
        nargs="*",
        default=[
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Race_x_SES",
            "Race_x_gender",
            "Religion",
            "SES",
            "Sexual_orientation",
        ],
        help="Categories: -a item1 item22",
    )
    parser.add_argument("-n", "--num_rows", type=int, default=-1, help="Number of rows to process")
    parser.add_argument(
        "-s", "--small_dataset", action="store_true", help="Use small dataset for evaluation"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Records between checkpoints (default: 50)",
    )
    parser.add_argument(
        "--main_concurrency",
        type=int,
        default=20,
        help="Main model concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--parser_concurrency",
        type=int,
        default=40,
        help="Parser model concurrent requests (default: 40)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Sliding window size for memory efficiency (default: 100)",
    )
    parser.add_argument(
        "--use-categories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate per-category metrics (2 files) vs overall-only (1 file)",
    )
    args = parser.parse_args()
    categories = args.categories
    logging.info(f"Performing BBQ evaluation on following categories: {categories}")

    config_file = args.config
    output_dir = args.output_dir
    dataset = args.dataset
    num_rows = args.num_rows
    small_dataset = args.small_dataset
    resume = args.resume
    checkpoint_interval = args.checkpoint_interval
    main_concurrency = args.main_concurrency
    parser_concurrency = args.parser_concurrency
    window_size = args.window_size
    use_categories = args.use_categories

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    main_bbq(
        categories=categories,
        config_file=config_file,
        output_dir=output_dir,
        dataset=dataset,
        num_rows=num_rows,
        small_dataset=small_dataset,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        main_concurrency=main_concurrency,
        parser_concurrency=parser_concurrency,
        window_size=window_size,
        use_categories=use_categories,
    )


if __name__ == "__main__":
    main()
