use std::{collections::hash_map::Entry, net::SocketAddr, num::NonZeroU8};

use fxhash::FxHashMap;
use laminar::{Packet, Socket};
use unros::{
    pubsub::Subscriber,
    setup_logging,
    tokio::sync::oneshot::{self, error::TryRecvError},
    RuntimeContext,
};

use crate::{NetworkPeer, PeerQuirk, SpecialMessage};

pub struct NetworkPublisher {
    pub(crate) setter: Box<dyn Fn(Box<[u8]>) + Send + Sync>,
    pub(crate) valid: Box<dyn Fn() -> bool + Send + Sync>,
}

pub(super) enum AwaitingNegotiationReq {
    ServerNegotiation {
        negotiation_recv: oneshot::Receiver<FxHashMap<NonZeroU8, NetworkPublisher>>,
        client_negotiation_sender: oneshot::Sender<()>,
    },
    ServerAwaitNegotiateResponse {
        packets_router: FxHashMap<NonZeroU8, NetworkPublisher>,
        client_negotiation_sender: oneshot::Sender<()>,
    },
    ClientNegotiation {
        negotiation_recv: oneshot::Receiver<FxHashMap<NonZeroU8, NetworkPublisher>>,
    },
}

pub(super) enum PeerStateMachine {
    /// Variant only on the client side, created after the init data has been sent to the server
    /// but before a `Negotiate` has been received from the server.
    Connecting {
        peer_sender: oneshot::Sender<NetworkPeer>,
    },

    /// Variant on both the client and server side.
    ///
    /// Server - Server is waiting for code to negotiate channels internally before sending `Negotiate` to client.
    /// Waits for `Negotiate` from client before transitioning to `Connected`.  
    /// Client - Client is waiting for code to negotiate channels internally before sending `Negotiate` to server.
    /// Instantly transitions to `Connected` after sending `Negotiate`.
    AwaitingNegotiation {
        packets_sub: Subscriber<Packet>,
        req: AwaitingNegotiationReq,
    },

    /// Variant on both the client and server side.
    ///
    /// Channels have been negotiated so it is safe to send and receive packets.
    Connected {
        packets_sub: Subscriber<Packet>,
        packets_router: FxHashMap<NonZeroU8, NetworkPublisher>,
    },
}

#[derive(PartialEq, Eq)]
pub(super) enum Retention {
    Drop,
    Retain,
}

impl PeerStateMachine {
    pub fn provide_data(
        &mut self,
        packet: Packet,
        context: &RuntimeContext,
        peer_buffer_size: usize,
    ) -> Retention {
        let data = packet.payload();
        let addr = packet.addr();
        setup_logging!(context);

        match self {
            PeerStateMachine::Connecting { peer_sender } => {
                match bitcode::decode::<SpecialMessage>(data) {
                    Ok(SpecialMessage::Disconnect) => return Retention::Drop,
                    Ok(SpecialMessage::Negotiate) => {
                        let (packets_router_sender, packets_router_recv) = oneshot::channel();
                        let packets_sub = Subscriber::new(peer_buffer_size);
                        let peer = NetworkPeer {
                            remote_addr: addr,
                            packets_to_send: packets_sub.create_subscription(),
                            packets_router: packets_router_sender,
                            quirk: PeerQuirk::ClientSide,
                        };
                        let peer_sender = std::mem::replace(peer_sender, oneshot::channel().0);
                        *self = PeerStateMachine::AwaitingNegotiation {
                            packets_sub,
                            req: AwaitingNegotiationReq::ClientNegotiation {
                                negotiation_recv: packets_router_recv,
                            },
                        };
                        if peer_sender.send(peer).is_ok() {
                            Retention::Retain
                        } else {
                            Retention::Drop
                        }
                    }
                    Ok(SpecialMessage::Ack) => if peer_sender.is_closed() {
                        Retention::Drop
                    } else {
                        Retention::Retain
                    }
                    Err(e) => {
                        error!("Failed to parse special_msg from {addr}: {e}");
                        Retention::Retain
                    }
                }
            }

            PeerStateMachine::AwaitingNegotiation { req, packets_sub } => {
                match bitcode::decode::<SpecialMessage>(data) {
                    Ok(SpecialMessage::Disconnect) => return Retention::Drop,
                    Ok(SpecialMessage::Negotiate) => match req {
                        AwaitingNegotiationReq::ServerNegotiation { .. } => {
                            warn!("Unexpected Negotiate from {addr}");
                            Retention::Retain
                        }
                        AwaitingNegotiationReq::ServerAwaitNegotiateResponse {
                            packets_router,
                            client_negotiation_sender,
                        } => if std::mem::replace(client_negotiation_sender, oneshot::channel().0).send(()).is_ok() {
                            *self = PeerStateMachine::Connected {
                                packets_sub: std::mem::replace(packets_sub, Subscriber::new(1)),
                                packets_router: std::mem::take(packets_router),
                            };
                            Retention::Retain
                        } else {
                            Retention::Drop
                        }
                        AwaitingNegotiationReq::ClientNegotiation { .. } => {
                            warn!("Unexpected Negotiate from {addr}");
                            Retention::Retain
                        }
                    },
                    Ok(SpecialMessage::Ack) => if let AwaitingNegotiationReq::ServerAwaitNegotiateResponse { client_negotiation_sender, .. } = req {
                        if client_negotiation_sender.is_closed() {
                            Retention::Drop
                        } else {
                            Retention::Retain
                        }
                    } else {
                        Retention::Retain
                    },
                    Err(e) => {
                        error!("Failed to parse special_msg from {addr}: {e}");
                        Retention::Retain
                    }
                }
            }

            PeerStateMachine::Connected {
                packets_router,
                packets_sub: _,
            } => {
                let channel = *data.last().unwrap();
                let data = data.split_at(data.len() - 1).0;

                let Some(channel) = NonZeroU8::new(channel) else {
                    match bitcode::decode::<SpecialMessage>(data) {
                        Ok(SpecialMessage::Disconnect) => return Retention::Drop,

                        Ok(x) => error!("Unexpected special_msg from {addr}: {x:?}"),
                        Err(e) => error!("Failed to parse special_msg from {addr}: {e}"),
                    }
                    return Retention::Retain;
                };

                match packets_router.entry(channel) {
                    Entry::Occupied(entry) => {
                        let publisher = entry.get();
                        if (publisher.valid)() {
                            (publisher.setter)(data.into());
                        } else {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(_) => {
                        error!("Unrecognized channel: {}", channel);
                    }
                }

                Retention::Retain
            }
        }
    }

    pub fn poll(
        &mut self,
        socket: &mut Socket,
        addr: SocketAddr,
        context: &RuntimeContext,
    ) -> Retention {
        setup_logging!(context);

        match self {
            PeerStateMachine::Connecting { peer_sender } => {
                if peer_sender.is_closed() {
                    Retention::Drop
                } else {
                    Retention::Retain
                }
            }
            PeerStateMachine::AwaitingNegotiation { req, packets_sub } => match req {
                AwaitingNegotiationReq::ServerNegotiation {
                    negotiation_recv,
                    client_negotiation_sender,
                } => match negotiation_recv.try_recv() {
                    Ok(packets_router) => {
                        if let Err(e) = socket.send(Packet::reliable_ordered(
                            addr,
                            bitcode::encode(&SpecialMessage::Negotiate).unwrap(),
                            None,
                        )) {
                            error!("Failed to send Negotiate to {addr}: {e}");
                        }

                        *req = AwaitingNegotiationReq::ServerAwaitNegotiateResponse {
                            packets_router,
                            client_negotiation_sender: std::mem::replace(client_negotiation_sender, oneshot::channel().0),
                        };

                        Retention::Retain
                    }
                    Err(TryRecvError::Closed) => Retention::Drop,
                    Err(TryRecvError::Empty) => Retention::Retain,
                },
                AwaitingNegotiationReq::ServerAwaitNegotiateResponse {
                    packets_router: _,
                    client_negotiation_sender,
                } => if client_negotiation_sender.is_closed() {
                    Retention::Drop
                } else {
                    Retention::Retain
                }
                AwaitingNegotiationReq::ClientNegotiation { negotiation_recv } => match negotiation_recv.try_recv() {
                    Ok(packets_router) => {
                        if let Err(e) = socket.send(Packet::reliable_ordered(
                            addr,
                            bitcode::encode(&SpecialMessage::Negotiate).unwrap(),
                            None,
                        )) {
                            error!("Failed to send Negotiate to {addr}: {e}");
                        }
                        
                        *self = PeerStateMachine::Connected {
                            packets_sub: std::mem::replace(packets_sub, Subscriber::new(1)),
                            packets_router,
                        };

                        Retention::Retain
                    }
                    Err(TryRecvError::Closed) => Retention::Drop,
                    Err(TryRecvError::Empty) => Retention::Retain,
                }
            },
            PeerStateMachine::Connected {
                packets_router,
                packets_sub,
            } => {
                while let Some(packet) = packets_sub.try_recv() {
                    if let Err(e) = socket.send(packet) {
                        error!("Failed to send packet to {addr}: {e}");
                    }
                }
                if packets_router.is_empty() && packets_sub.get_pub_count() == 0 {
                    Retention::Drop
                } else {
                    Retention::Retain
                }
            }
        }
    }
}