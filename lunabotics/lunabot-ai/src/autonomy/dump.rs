use ares_bt::{action::AlwaysFail, branching::IfElse, Behavior, Status};

use crate::{blackboard::LunabotBlackboard, Action};

use super::{Autonomy, AutonomyStage};

pub(super) fn dump() -> impl Behavior<LunabotBlackboard, Action> {
    IfElse::new(
        |blackboard: &mut LunabotBlackboard| {
            matches!(
                blackboard.get_autonomy(),
                Autonomy::FullAutonomy(AutonomyStage::Dump)
                    | Autonomy::PartialAutonomy(AutonomyStage::Dump)
            )
            .into()
        },
        |blackboard: &mut LunabotBlackboard| {
            blackboard.get_autonomy().advance();
            Status::Success
        },
        AlwaysFail,
    )
}