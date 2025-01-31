import click

@click.command()
def main():
    # we take in:
    # - a checkpoint file
    # - an hdf5 path
    # - output path
    # - optional: num samples (if none we just do the whole thing), batch size (use a sensible default)
    #             
    # no wait this is silly. rethink.
    # use lightning to predict params and dump to file. then this script just takes a 
    # param file (and optionally the dataset file used to generate the params)
    # and outputs the audio. this way we can just run on cpu
    pass

if __name__ == "__main__":
    main()
