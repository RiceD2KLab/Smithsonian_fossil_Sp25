from src.modeling.detr.detr_utils import (
    initialize_model,
    CocoDetectionTransform,
    collate_fn,
    get_optimizer_scheduler,
    train_model
)
from scripts.config_extractor import load_config as project_load_config
from src.modeling.detr.config_extractor import load_config as detr_load_config
import torch
from torch.utils.data import DataLoader


def main(config_path="../detr/detr_config.json"):
    # Step 1: Load config
    config = detr_load_config(config_path)
    project_config = project_load_config("../../config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 2: Create dataset
    train_dataset = CocoDetectionTransform(
        project_config["abs_path_to_ndpi_tiles_dir"],
        "../detr/tmp/pollen_train.json"
    )

    # Step 3: Create Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Step 4: Load model + processor
    model, processor = initialize_model(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        weights_path=None,
        device=device
    )

    # Step 5: Train model
    opt, sched = get_optimizer_scheduler(
        model=model,
        train_loader=train_loader,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        epochs=config["num_epochs"],
        warmup_frac=config["warmup_fraction"],
    )
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=opt,
        scheduler=sched,
        device=device,
        epochs=config["num_epochs"],
        clip_norm=config["max_grad_norm"],
        log_every=10
    )

    # Step 6: Save model
    torch.save(model.state_dict(), "../detr/detr_pollen.pth")


    print("Pipeline complete.")

if __name__ == "__main__":
    main()
