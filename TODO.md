# to do

- [ ] make the history in the pipeline object more fine grained (i.e. track the time of individual methods as opposed to just module.main)

- [November 2016] be a bit more sensible when backing up models (especially discriminative models), e.g., using gzip formats to reduce space (almost completed with ZubrSerializable, but not implemented into all modelsyet).

-- remove pickling of objects. Given the design of the tool, objects need to be backed up and built during runtime, which I've tended to do by pickling entire objects, then reloading. This is okay for small datasets and models, but it becomes prohibitively slow and memory intensive larger ones. This can be overcome by building a custom backup protocol, in some classes this is called ``backup`` and ``load_backup``, that uses, among other things, numpy's savez_compressed.
