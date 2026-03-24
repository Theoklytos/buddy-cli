import os
import sys
def list_files(directory):
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    files = list_files(directory)
    with open('all_code.txt', 'w') as outfile:
        for f in files:
            filepath = os.path.join(directory, f)
            outfile.write(f'### {f}\n')
            try:
                with open(filepath, 'r') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f'Error reading file {f}: {str(e)}\n')