use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{atomic::Ordering, Weak},
};

use log::warn;

use super::SubscriberInner;

pub struct PublisherToken(pub(super) ());

pub trait Subscription {
    type Item;

    fn push(&mut self, value: Self::Item) -> bool;

    /// Changes the generic type of this `Subscription` using the given `map` function.
    fn map<F, O>(self, map: F) -> Map<Self, F, O>
    where
        Self: Sized,
        F: FnMut(O) -> Self::Item,
    {
        Map {
            inner: self,
            map,
            _phantom: PhantomData,
        }
    }

    /// Changes the generic type of this `Subscription` using the given `filter_map` function.
    ///
    /// If the function returns `None`, the value will not be published.
    fn filter_map<F, O>(self, map: F) -> FilterMap<Self, F, O>
    where
        Self: Sized,
        F: FnMut(O) -> Self::Item,
    {
        FilterMap {
            inner: self,
            map,
            _phantom: PhantomData,
        }
    }

    fn boxed(self) -> BoxedSubscription<Self::Item>
    where
        Self: Sized + Send + 'static,
    {
        Box::new(self)
    }

    // fn zip<V: 'static>(mut self, mut other: DirectSubscription<V>) -> DirectSubscription<(T, V)> where Self: Sized {
    // self.pub_count.append(&mut other.pub_count);
    // DirectSubscription {
    //     queue: Box::new(move |(left, right)| {
    //         let left_result = self.queue.push(left);
    //         let right_result = other.queue.push(right);
    //         match left_result {
    //             EnqueueResult::Ok => right_result,
    //             EnqueueResult::Full => {
    //                 if right_result == EnqueueResult::Closed {
    //                     EnqueueResult::Closed
    //                 } else {
    //                     EnqueueResult::Full
    //                 }
    //             }
    //             EnqueueResult::Closed => EnqueueResult::Closed,
    //         }
    //     }),
    //     notify: self.notify,
    //     lag: 0,
    //     name: None,
    //     pub_count: self.pub_count,
    // }
    // }

    /// Provides a name to this subscription, which enables lag logging.
    ///
    /// If the `Publisher` that accepts this `Subscription` cannot push
    /// new messages into this `Subscription` without deleting old message,
    /// we say that the `Subscription` is lagging. Catching lagging is important
    /// as it indicates data loss and a lack of processing speed. With a name,
    /// these lags will be logged as warnings in the standard log file (`.log`).
    #[must_use]
    fn set_name(mut self, name: impl Into<String>) -> Self
    where
        Self: Sized,
    {
        self.set_name_mut(name.into().into_boxed_str());
        self
    }

    fn set_name_mut(&mut self, name: Box<str>);

    fn increment_publishers(&self, token: PublisherToken);
    fn decrement_publishers(&self, token: PublisherToken);
}

/// An object that must be passed to a `Publisher`, enabling the `Subscriber`
/// that created the subscription to receive messages from that `Publisher`.
///
/// If dropped, no change will occur to the `Subscriber` and no resources will be leaked.
pub struct DirectSubscription<T> {
    pub(super) sub: Weak<SubscriberInner<T>>,
    pub(super) name: Option<Box<str>>,
    pub(super) lag: usize,
}

impl<T> Clone for DirectSubscription<T> {
    fn clone(&self) -> Self {
        Self {
            sub: self.sub.clone(),
            name: self.name.clone(),
            lag: self.lag.clone(),
        }
    }
}

impl<T> Subscription for DirectSubscription<T> {
    type Item = T;

    fn push(&mut self, value: Self::Item) -> bool {
        if let Some(sub) = self.sub.upgrade() {
            if sub.queue.force_push(value).is_some() {
                self.lag += 1;
                if let Some(name) = &self.name {
                    warn!(target: "publishers", "{name} lagging by {} messages", self.lag);
                }
            } else {
                self.lag = 0;
                sub.notify.notify_one();
            }
            true
        } else {
            false
        }
    }

    fn set_name_mut(&mut self, name: Box<str>)
    where
        Self: Sized,
    {
        self.name = Some(name);
    }

    fn increment_publishers(&self, _token: PublisherToken) {
        if let Some(sub) = self.sub.upgrade() {
            sub.pub_count.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn decrement_publishers(&self, _token: PublisherToken) {
        if let Some(sub) = self.sub.upgrade() {
            sub.pub_count.fetch_sub(1, Ordering::AcqRel);
            sub.notify.notify_one();
        }
    }
}

pub struct Map<I, F, O> {
    inner: I,
    map: F,
    _phantom: PhantomData<O>,
}

impl<O, I, F> Subscription for Map<I, F, O>
where
    I: Subscription,
    F: FnMut(O) -> I::Item,
{
    type Item = O;

    fn push(&mut self, value: Self::Item) -> bool {
        self.inner.push((self.map)(value))
    }

    fn set_name_mut(&mut self, name: Box<str>) {
        self.inner.set_name_mut(name);
    }

    fn increment_publishers(&self, token: PublisherToken) {
        self.inner.increment_publishers(token);
    }

    fn decrement_publishers(&self, token: PublisherToken) {
        self.inner.decrement_publishers(token);
    }
}

impl<I: Clone, F: Clone, O> Clone for Map<I, F, O> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            map: self.map.clone(),
            _phantom: PhantomData,
        }
    }
}

pub struct FilterMap<I, F, O> {
    inner: I,
    map: F,
    _phantom: PhantomData<O>,
}

impl<O, I, F> Subscription for FilterMap<I, F, O>
where
    I: Subscription,
    F: FnMut(O) -> Option<I::Item>,
{
    type Item = O;

    fn push(&mut self, value: Self::Item) -> bool {
        if let Some(value) = (self.map)(value) {
            self.inner.push(value)
        } else {
            true
        }
    }

    fn set_name_mut(&mut self, name: Box<str>) {
        self.inner.set_name_mut(name);
    }

    fn increment_publishers(&self, token: PublisherToken) {
        self.inner.increment_publishers(token);
    }

    fn decrement_publishers(&self, token: PublisherToken) {
        self.inner.decrement_publishers(token);
    }
}

impl<I: Clone, F: Clone, O> Clone for FilterMap<I, F, O> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            map: self.map.clone(),
            _phantom: PhantomData,
        }
    }
}

pub type BoxedSubscription<T> = Box<dyn Subscription<Item = T> + Send>;

impl<T> Subscription for BoxedSubscription<T> {
    type Item = T;

    fn push(&mut self, value: Self::Item) -> bool {
        self.deref_mut().push(value)
    }

    fn set_name_mut(&mut self, name: Box<str>) {
        self.deref_mut().set_name_mut(name);
    }

    fn increment_publishers(&self, token: PublisherToken) {
        self.deref().increment_publishers(token);
    }

    fn decrement_publishers(&self, token: PublisherToken) {
        self.deref().decrement_publishers(token);
    }
}