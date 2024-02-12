import torch
import torch.nn.functional as F

from distributed.gather import all_reduce_mean_grad
from schedules import schedule_from_kwargs
from utils.multi_crop_utils import multi_crop_loss
from .base.schedulable_loss_base import SchedulableLossBase


class DinoLoss(SchedulableLossBase):
    def __init__(self, teacher_temperature=None, teacher_temperature_schedule=None, **kwargs):
        super().__init__(**kwargs)
        assert (teacher_temperature is not None) ^ (teacher_temperature_schedule is not None)
        self.teacher_temperature = teacher_temperature
        self.teacher_temperature_schedule = schedule_from_kwargs(
            teacher_temperature_schedule,
            update_counter=self.update_counter,
        )

    def forward(self, student, teacher, center, center_update_info):
        assert len(teacher) == 2
        # teacher centering and sharpening
        if self.teacher_temperature is not None:
            teacher_temp = self.teacher_temperature
        else:
            teacher_temp = self.teacher_temperature_schedule.get_value(self.update_counter.cur_checkpoint)
        # concat for softmax/center update
        teacher_chunks = len(teacher)
        teacher = torch.concat(teacher)
        teacher = F.softmax((teacher - center) / teacher_temp, dim=-1)
        # populate update_center_info (center is updated only after update step to allow gradient accumulation)
        with torch.no_grad():
            batch_center = torch.sum(teacher, dim=0, keepdim=True)
            batch_center = all_reduce_mean_grad(batch_center)
            batch_center = batch_center / len(teacher)
            # https://github.com/facebookresearch/dino/blob/main/main_dino.py#L364
            center_update_info["momentum"] = 0.9
            center_update_info["batch_centers"].append(batch_center)
        # back to list
        teacher = teacher.chunk(teacher_chunks)

        losses = multi_crop_loss(student, teacher, self._forward)
        return {f"view{i}-view{j}": loss for (i, j), loss in losses.items()}

    @staticmethod
    def _forward(student, teacher):
        # https://github.com/facebookresearch/dino/blob/main/main_dino.py#L364
        student_temperature = 0.1
        student = student / student_temperature

        # TODO log teacher_temp
        # TODO check if teacher has grad (shouldn't have)

        loss = torch.sum(-teacher * F.log_softmax(student, dim=-1), dim=-1)
        return loss.mean()

