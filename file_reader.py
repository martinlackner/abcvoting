

def read_soi_file(filename, max_approval_percent=0.8):
	with open(filename, "r") as f:
		line = f.readline()
		candidate_count = int(line.strip())
		candidate_map = {}
		for i in range(candidate_count):
			parts = f.readline().strip().split(",")
			candidate_map[int(parts[0].strip())] = parts[1].strip()

		parts = f.readline().split(",")
		# voter_count = int(parts[0].strip())
		# voter_sum = int(parts[1].strip())
		unique_orders = int(parts[2].strip())
		unique_candidates = set()
		profile = []
		for i in range(unique_orders):
			line = f.readline().strip()
			count = int(line[:line.index(",")])
			parts = line[line.index(",")+1:].strip().split(",")
			vote = []
			take_cands = min(len(parts), int(len(parts)*max_approval_percent))
			for pos in range(max(1, take_cands)):
				vote.append(int(parts[pos].strip()))
				unique_candidates.add(parts[pos].strip())
			for _ in range(count):
				profile.append(vote)

		used_candidate_count = len(unique_candidates)

		if len(profile) != unique_orders:
			print("Error: Missing unique rankings")

		return candidate_map, profile, used_candidate_count
