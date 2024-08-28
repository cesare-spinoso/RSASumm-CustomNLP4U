from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from src import SCRATCH_CACHE_DIR
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch

from src.finetuning.literal_summarizer.models import LiteralSummarizerPLModule

# with init_empty_weights():
#     bart = BartForConditionalGeneration.from_pretrained(
#         "facebook/bart-large", cache_dir=SCRATCH_CACHE_DIR
#     )

# bart.tie_weights()

# checkpoint_path = "/network/scratch/c/cesare.spinoso/rsasumm/finetuning/2024-08-06-17:38:51.477512/lightning_logs/version_0/checkpoints/epoch=43-step=4972.ckpt"

# # Load only the state_dict from the PyTorch Lightning checkpoint
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# state_dict = checkpoint["state_dict"]  # Extract the model's state_dict

# model = load_checkpoint_and_dispatch(
#     model=bart, checkpoint=checkpoint_path, device_map="auto"
# )

checkpoint_path = "/network/scratch/c/cesare.spinoso/rsasumm/finetuning/2024-08-06-17:38:51.477512/lightning_logs/version_0/checkpoints/epoch=43-step=4972.ckpt"


pl_module = LiteralSummarizerPLModule.load_from_checkpoint(checkpoint_path)
model = pl_module.model
model = load_checkpoint_and_dispatch(
    model=model, checkpoint=checkpoint_path, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large", cache_dir=SCRATCH_CACHE_DIR
)
input_ids = tokenizer(
    [
        """
        In the coming days, President Trump Donald John TrumpFive takeaways from the Democratic National Convention What we'll remember from the 2020 Biden convention Chris Wallace labels Biden's acceptance speech 'enormously effective' MORE plans to announce his final decision on whether the United States will withdraw from the 2015 Iran nuclear accord. President Trump, who has described the agreement as “one of the worst deals” he has ever witnessed, is expected to leave the pact.\n\nUltimately, this is the right decision for the United States and for global security at large. From the outset, the accord, which provided sanctions relief for Iran in exchange for restrictions on their nuclear program, has a number of fatal flaws.\n\nADVERTISEMENT\n\nForemost, while the plan limits Iran’s access to uranium, this restriction only lasts until 2025 to 2030. After that, the Iranians are free to revitalize their nuclear program on a potentially even larger scale. Notably, however, this “sunset” clause, which allow parts of the deal to expire, are the least of the deal’s shortcomings.\n\nOne of the primary failures of the deal is that the agreement fails to address Iran’s ballistic missile program. As such, the country has continued to unrestrictedly build and test ballistic missiles. Moreover, President Trump and others have rightly objected to the terms under which regulatory inspectors are permitted to visit nuclear sites.\n\nThe terms of the deal give Iran 14 days to object to a request for inspection, followed by a period of seven days for an arbitration committee to rule about the inspection, and another three days for Tehran to set up an inspection. Thus, this provides Iran with up to 24 days to conceal, destroy, or relocate contraband materials.\n\nEven more problematically, Iran has stated that it will prohibit inspections of military sites, thus further complicating the issue of compliance verification. These flaws have become so glaringly problematic that even those who once championed the deal have begun to question it.\n\n“Everyone recognizes that the deal is not ideal. I think President Obama would say the deal is not ideal,” said Bob Einhorn, who was the State Department’s special adviser for nonproliferation and arms control during the Obama administration. While these flaws are not necessarily brand new, there have been several recent alarming developments that have sparked concern among elected officials.\n\nLast week, standing in front a screen which blatantly displayed the text “Iran lied” in all caps, Israeli Prime Minister Benjamin Netanyahu declared that Israeli intelligence services had obtained proof that Iran had been deceptive about its nuclear program.\n\nNetanyahu claims to have 55,000 pages and 183 CDs full of evidence that Iran had sought “to design, produce and test five warheads with 10 kiloton of TNT yield for integration on missiles.”
        """
    ],
    return_tensors="pt",
).to("cuda")
outputs = model.generate(**input_ids, max_length=100, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
