use common::LunabotStage;
use log::{error, info};
use luna_bt::{Behaviour, ERR, OK};
use nalgebra::Isometry3;

use crate::{
    Autonomy, AutonomyStage, DriveComponent, LunabotBlackboard, PathfinderComponent,
    TeleOpComponent,
};

pub(super) fn traverse_obstacles<D, P, O, T>(
) -> Behaviour<'static, Option<LunabotBlackboard<D, P, O, T>>>
where
    D: DriveComponent,
    P: PathfinderComponent,
    O: Fn() -> Isometry3<f64>,
    T: TeleOpComponent,
{
    Behaviour::if_else(
        Behaviour::action(|bb: &mut Option<LunabotBlackboard<D, P, O, T>>| {
            let Some(bb) = bb else {
                error!("Blackboard is missing in traverse_obstacles");
                return ERR;
            };
            match bb.autonomy {
                Autonomy::PartialAutonomy(AutonomyStage::TraverseObstacles) => OK,
                Autonomy::FullAutonomy(AutonomyStage::TraverseObstacles) => OK,
                _ => ERR,
            }
        }),
        Behaviour::sequence([Behaviour::action(
            |bb: &mut Option<LunabotBlackboard<D, P, O, T>>| {
                let bb = bb.as_mut().unwrap();
                bb.teleop.set_lunabot_stage(LunabotStage::TraverseObstacles);
                info!("Traversing obstacles");
                std::thread::sleep(std::time::Duration::from_secs(1));
                if let Autonomy::PartialAutonomy(_) = bb.autonomy {
                    bb.autonomy = Autonomy::None;
                }
                OK
            },
        )]),
        Behaviour::constant(ERR),
    )
}
