
import java.io.*;
import java.util.*;

public class TOPK_PSO {

    private Map<Integer, Integer> itemTWU = new HashMap<>();
    private List<List<Pair>> database = new ArrayList<>();
    private Particle gBest;
    private Particle[] pBest;
    private Particle[] population;
    private int maxTransactionLength = 0;
    private ArrayList<Item> items = new ArrayList<>();
    private HashSet<BitSet> explored = new HashSet<>();
    private HashMap<Integer, Integer> itemNamesRev = new HashMap<>();
    private int shortestTransactionID;
    private int std;
    private boolean avgEstimate;
    private int lowEst = 0;
    private int highEst = 0;
    int minSolutionFitness = 0;
    Solutions sols;
    int count = 0;
    boolean newS = false;
    long disUtil = 0;
    int minUtil;
    //ArrayList<Integer> it = new ArrayList<>();
    //ArrayList<Integer> pat = new ArrayList<>();


    //file paths
    final String dataset = "chess";
    final String dataPath = "C:\\Users\\homse\\OneDrive\\Desktop\\datasets\\" + dataset + ".txt"; //input file path
    final String resultPath = "C:\\Users\\homse\\OneDrive\\Desktop\\datasets\\out.txt"; //output file path
    final String convPath = "D:\\Documents\\Skole\\Master\\Experiments\\" + dataset + "\\";

    //Algorithm parameters
    final int pop_size = 20; // the size of the population
    final int iterations = 10000; // the number of iterations before termination
    final int k = 50;
    final boolean closed = false; //true = find CHUIS, false = find HUIS
    final boolean runPrune = false;


    //stats
    double maxMemory = 0; // the maximum memory usage
    long startTimestamp = 0; // the time the algorithm started
    long endTimestamp = 0; // the time the algorithm terminated
    long totalUtil = 0; // the total utility in the DB


    // this class represent an item and its utility in a transaction
    private static class Pair implements Comparable<Pair> {
        int item;
        int utility;

        public Pair(int item, int utility) {
            this.item = item;
            this.utility = utility;
        }

        @Override
        public int compareTo(Pair o) {
            return (this.item < o.item) ? -1 : 1;
        }
    }

    private static class Item {
        int item; //item-name
        BitSet TIDS; //TID set
        int totalUtil = 0; //utility of the item
        int avgUtil; // average utility of item
        int maxUtil = 0; // maximum utility of item

        public Item(int item) {
            TIDS = new BitSet();
            this.item = item;
        }

        @Override
        public String toString() {
            return String.valueOf(item);
        }
    }

    // this class represent a particle
    private static class Particle implements Comparable<Particle> {
        BitSet X; // items contained in particle (encoding vector)
        int fitness; // fitness/utility of particle
        int estFitness; // estimated fitness of particle

        public Particle(int size) {
            this.X = new BitSet(size);
        }

        public Particle(BitSet bitset, int fitness) {
            this.X = (BitSet) bitset.clone();
            this.fitness = fitness;
        }

        @Override
        public String toString() {
            return String.valueOf(fitness);
        }

        @Override
        public int compareTo(Particle o) {
            if (this.fitness == o.fitness) {
                return 0;
            }
            return (this.fitness < o.fitness) ? -1 : 1;
        }
    }

    public class Solutions {
        TreeSet<Particle> sol = new TreeSet<>();
        int size;

        public Solutions(int size) {
            this.size = size;
        }

        public void add(Particle p) {
            if (sol.size() == size) {
                sol.pollFirst(); //TODO: Something wrong, verify by summing list at end (chess, k=50, avgEST)
            }
            sol.add(p);
            newS = true;
        }

        public TreeSet<Particle> getSol() {
            return sol;
        }

        public int getMin() {
            return sol.first().fitness;
        }
    }

    /**
     * Call this method to run the algorithm. File paths and algorithm parameters must be set in top of this class
     *
     * @throws IOException
     */
    public void run() throws IOException {
        maxMemory = 0;
        startTimestamp = System.currentTimeMillis();
        sols = new Solutions(k);
        init(); //reads input file and prunes DB


        System.out.println("TWU_SIZE: " + items.size());
        checkMemory();


        //calculate average utility of each item and find the standard deviation between avgUtil & maxUtil
        std = 0; // the standard deviation
        for (Item item : items) {
            item.avgUtil = 1 + (item.totalUtil / item.TIDS.cardinality());
            std += item.maxUtil - item.avgUtil;
        }

        if (!items.isEmpty()) {
            std = std / items.size();
            //only use avgEstimates if the standard deviation is small compared to the minUtil
            //avgEstimate = (double) std / minUtil < 0.0001;
            avgEstimate = true;
            //initialize the population
            generatePop();
            List<Double> probRange = rouletteProbKHUI(); //roulette probabilities for current discovered HUIs
            for (int i = 0; i < iterations; i++) {
                newS = false;
                //update each particle in population
                update();

                //gBest update RWS
                if (minSolutionFitness >= minUtil) {
                    if (newS) { //new solutions are discovered, probability range must be updated
                        probRange = rouletteProbKHUI();
                    }
                    int pos = rouletteSelect(probRange);
                    selectGBest(pos);
                }

                if (newS) {
                    System.out.println("iteration: " + i);
                }


                if (i % 100 == 0 && highEst > 0 && i > 0) { //check each 100th iteration
                    //Tighten std if mostly overestimates are made (only relevant when avgEstimates is active)
                    std = ((double) lowEst / highEst < 0.01) ? 1 : std;
                }

            }
        }
        endTimestamp = System.currentTimeMillis();
        checkMemory();
        writeOut();
        System.out.println(sols.getSol());
        System.out.println("skipped: " + count);
        //writeRes();
    }


    private void generatePop() { //TODO: can be more effective when creating already explored particles.
        List<Double> rouletteProbabilities = rouletteProbabilities();
        population = new Particle[pop_size];
        pBest = new Particle[pop_size];
        for (int i = 0; i < pop_size; i++) {
            //k is the number of items to include in the particle
            int k = (int) (Math.random() * maxTransactionLength) + 1;
            //j is the current number of items that has been included
            int j = 0;
            Particle p = new Particle(items.size());
            while (j < k) {
                //select the item
                int pos = rouletteSelect(rouletteProbabilities);
                //if item is not previously selected, select it and increment j.
                if (!p.X.get(items.get(pos).item)) {
                    p.X.set(items.get(pos).item);
                    j++;
                }
            }
            BitSet tidSet = pev_check(p); //transactions the particle occur
            p.fitness = calcFitness(p, tidSet, -1);
            population[i] = p;
            pBest[i] = new Particle(p.X, p.fitness); //initialize pBest
            if (!explored.contains(p.X)) {
                //check if HUI/CHUI
                if (p.fitness > minSolutionFitness) {
                    if (closed) {
                        //check if particle is closed
                        if (isClosed(p, shortestTransactionID, tidSet)) {
                            Particle s = new Particle(p.X, p.fitness);
                            sols.add(s);
                            minSolutionFitness = sols.getMin();
                        }
                    } else {
                        Particle s = new Particle(p.X, p.fitness);
                        sols.add(s);
                        minSolutionFitness = sols.getMin();
                    }
                }
            }
            if (i == 0) {
                gBest = new Particle(p.X, p.fitness); //initialize gBest
            } else {
                if (p.fitness > gBest.fitness) {
                    gBest = new Particle(p.X, p.fitness); //update gBest
                }
            }
            BitSet clone = (BitSet) p.X.clone();
            explored.add(clone); //set particle as explored
        }
    }

    /**
     * The pev-check verifies that the particle exists in the database and modifies it if not.
     * Furthermore, it calculates the avg/max fitness estimate and returns the TidSet of the particle.
     *
     * @param p The particle
     * @return orgBitSet: The transactions the itemset of the particle occur (TidSet)
     */
    private BitSet pev_check(Particle p) {
        int item1 = p.X.nextSetBit(0);
        if (item1 == -1) {
            return null;
        }
        BitSet orgBitSet = (BitSet) items.get(item1 - 1).TIDS.clone();
        BitSet copyBitSet = (BitSet) orgBitSet.clone();
        p.estFitness = avgEstimate ? items.get(item1 - 1).avgUtil : items.get(item1 - 1).maxUtil;
        for (int i = p.X.nextSetBit(item1 + 1); i != -1; i = p.X.nextSetBit(i + 1)) {
            orgBitSet.and(items.get(i - 1).TIDS);
            //the two items have common transactions
            if (orgBitSet.cardinality() > 0) {
                copyBitSet = (BitSet) orgBitSet.clone();
                p.estFitness += avgEstimate ? (items.get(i - 1).avgUtil) : (items.get(i - 1).maxUtil);
            } else {
                // no common transactions, remove the current item from the particle
                orgBitSet = (BitSet) copyBitSet.clone();
                p.X.clear(i);
            }
        }
        return orgBitSet;
    }

    /**
     * Verifies the closure of a particle
     *
     * @param p                   The particle
     * @param shortestTransaction The ID of the shortest transaction the particle occur, Stored during fitness calc
     * @param tidSet              the TIDSET of the particle
     * @return True if Closed, false otherwise
     */
    private boolean isClosed(Particle p, int shortestTransaction, BitSet tidSet) {
        int support = tidSet.cardinality();
        List<Pair> sTrans = database.get(shortestTransaction);
        //Loop all items in the shortest transaction
        for (Pair pair : sTrans) {
            //The item does not appear in the particle
            if (!p.X.get(pair.item)) {
                BitSet currentTids = (BitSet) tidSet.clone();
                BitSet newItem = (BitSet) items.get(pair.item - 1).TIDS.clone();
                currentTids.and(newItem);
                //The support of the particle is the same with the new item appended -> The particle is not closed
                if (currentTids.cardinality() == support) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Calculates the fitness of a particle
     *
     * @param p      The particle
     * @param tidSet TidSet of the particle
     * @param idx    The position of the particle in the population (to reference pBest), set to -1 if first population
     * @return The fitness of the particle
     */
    private int calcFitness(Particle p, BitSet tidSet, int idx) {
        int fitness = 0;
        int min = maxTransactionLength; //used to find the shortest transaction
        if (tidSet == null) {
            return 0; // particle does not occur in any transaction
        }

        //The particle only contains 1 item, return the fitness calculated during pre-processing
        if (p.X.cardinality() == 1) {
            return items.get(p.X.nextSetBit(0) - 1).totalUtil;
        }

        //estimate the fitness
        int support = tidSet.cardinality();
        int est = p.estFitness * support;
        int buffer = avgEstimate ? (std * support) : 0;
        if (idx != -1) {
            if (est + buffer < minSolutionFitness && est < pBest[idx].fitness) {
                // Skip fitness calculation
                return 0;
            }
        }

        //calculate exact fitness
        for (int i = tidSet.nextSetBit(0); i != -1; i = tidSet.nextSetBit(i + 1)) {
            int q = 0; //current index in transaction
            int item = p.X.nextSetBit(0);
            if (database.get(i).size() < min) {
                shortestTransactionID = i;
            }
            while (item != -1) {
                if (database.get(i).get(q).item == item) {
                    fitness += database.get(i).get(q).utility;
                    item = p.X.nextSetBit(item + 1);
                }
                q++;
            }
        }

        //Update overestimates and underestimates
        if (est + buffer < fitness) {
            lowEst++;
        } else {
            highEst++;
        }
        return fitness;
    }


    /**
     * Updates population and checks for new CHUIs
     */
    private void update() { //TODO: prune check TWU for new items
        for (int i = 0; i < pop_size; i++) {
            Particle p = population[i];
            //different bits between pBest and current particle
            List<Integer> diffList = bitDiff(pBest[i], p);
            //change a random amount of these bits
            changeParticle(diffList, i);
            //repeat for gBest
            diffList = bitDiff(gBest, p);
            changeParticle(diffList, i);


            if (explored.contains(p.X)) {
                //the particle is already explored, change one random bit
                int rand = (int) (items.size() * Math.random());
                int change = items.get(rand).item;
                p.X.flip(change);
            }

            //avoid PEV-check and fit. calc. if particle is already explored
            if (!explored.contains(p.X)) {
                //bitset before pev
                BitSet copy1 = (BitSet) p.X.clone();
                BitSet tidSet = pev_check(p);

                //check if explored again because pev_check can change the particle
                if (!explored.contains(p.X)) {
                    p.fitness = calcFitness(p, tidSet, i);
                    //update pBest and gBest
                    if (p.fitness > pBest[i].fitness) {
                        pBest[i] = new Particle(p.X, p.fitness);
                        if (p.fitness > gBest.fitness) {
                            gBest = new Particle(p.X, p.fitness);
                        }
                    }
                    if (p.fitness >= minSolutionFitness) {
                        if (closed) {
                            if (isClosed(p, shortestTransactionID, tidSet)) {
                                //Particle is CHUI
                                Particle s = new Particle(p.X, p.fitness);
                                sols.add(s);
                                minSolutionFitness = sols.getMin();
                            }
                        } else {
                            // particle is HUI
                            Particle s = new Particle(p.X, p.fitness);
                            sols.add(s);
                            minSolutionFitness = sols.getMin();
                        }
                    }
                    //bitset after pev
                    BitSet copy2 = (BitSet) p.X.clone();
                    explored.add(copy2); //set current particle as explored
                }
                explored.add(copy1); //set particle before PEV-check as explored
            }
        }
    }

    /**
     * Flips a random number of bits in current particle, only bits that are opposite to pBest/gBest are considered
     *
     * @param diffList bit differences between particle and pBest/gBest
     * @param pos      position of current particle in population
     */
    private void changeParticle(List<Integer> diffList, int pos) {
        //number of items so change
        int num = (int) (diffList.size() * Math.random() + 1);
        if (diffList.size() > 0) {
            for (int i = 0; i < num; i++) {
                //position to change
                int change = (int) (diffList.size() * Math.random());
                //flip the bit of the selected item
                population[pos].X.flip(diffList.get(change));
            }
        }
    }

    /**
     * Computes the bit difference between current particle and pBest/gBest
     *
     * @param best pBest/gBest
     * @param p    the current particle
     * @return List containing the different bit positions
     */
    private List<Integer> bitDiff(Particle best, Particle p) {
        List<Integer> diffList = new ArrayList<>();
        BitSet temp = (BitSet) best.X.clone();
        temp.xor(p.X);
        for (int i = temp.nextSetBit(0); i != -1; i = temp.nextSetBit(i + 1)) {
            diffList.add(i);
        }
        return diffList;
    }


    /**
     * Creates a list of probabilities for roulette wheel selection based on item TWU-values
     *
     * @return List of probability ranges
     */
    private List<Double> rouletteProbabilities() {
        List<Double> probRange = new ArrayList<>();
        double twuSum = 0;
        double tempSum = 0;
        //sum the twu values for all 1-HTWUIs
        for (int i = 0; i < items.size(); i++) {
            int item = items.get(i).item;
            twuSum += itemTWU.get(item);
        }
        //Set probabilities based on TWU-proportion
        for (int i = 0; i < items.size(); i++) {
            int item = items.get(i).item;
            tempSum += itemTWU.get(item);
            double percent = tempSum / twuSum;
            probRange.add(percent);
        }
        return probRange;
    }

    /**
     * Select an item based on the Roulette wheel probabilities
     *
     * @param probRange list of probability ranges for each item
     * @return
     */
    private int rouletteSelect(List<Double> probRange) { //TODO: make binary search
        double rand = Math.random();
        if (rand <= probRange.get(0)) {
            return 0;
        }
        int pos = 0;
        for (int i = 1; i < probRange.size(); i++) {
            if (rand > probRange.get(i - 1) && rand <= probRange.get(i)) {
                pos = i;
                break;
            }
        }
        return pos;
    }

    private void selectGBest(int pos) {
        int c = 0;
        for (Particle p : sols.getSol()) {
            if (c == pos) {
                gBest = new Particle(p.X, p.fitness);
                break;
            }
            c++;
        }
    }


    private List<Double> rouletteProbKHUI() {
        double sum = 0;
        double tempSum = 0;
        List<Double> rouletteProbs = new ArrayList<>();
        for (Particle hui : sols.getSol()) {
            sum += hui.fitness;
        }
        for (Particle hui : sols.getSol()) {
            tempSum += hui.fitness;
            double percent = tempSum / sum;
            rouletteProbs.add(percent);
        }
        return rouletteProbs;
    }

    private void init2() {
        Map<Integer, Integer> itemTWU1 = new HashMap<>(); //holds current TWU-value for each item
        List<Integer> transUtils = new ArrayList<>(); //holds TU-value for each transaction
        List<List<Pair>> db = new ArrayList<>();
        List<List<Pair>> tempDb = new ArrayList<>();

        Map<Integer, Integer> itemUtil = new HashMap<>();

        String currentLine;
        try (BufferedReader data = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataPath)))) {
            //1st DB-Scan: calculate TWU value for each item
            while ((currentLine = data.readLine()) != null) {
                String[] split = currentLine.split(":");
                String[] items = split[0].split(" ");
                String[] utilities = split[2].split(" ");
                int transactionUtility = Integer.parseInt(split[1]);
                totalUtil += transactionUtility;
                transUtils.add(transactionUtility);
                List<Pair> transaction = new ArrayList<>();
                for (int i = 0; i < items.length; i++) {
                    int item = Integer.parseInt(items[i]);
                    int util = Integer.parseInt(utilities[i]);
                    Pair pair = new Pair(item, util);
                    transaction.add(pair);
                    Integer twu = itemTWU1.get(item);
                    twu = (twu == null) ? transactionUtility : twu + transactionUtility;
                    itemTWU1.put(item, twu);
                    //calculate utility of size 1 itemsets
                    Integer currUtil = itemUtil.get(item);
                    currUtil = (currUtil == null) ? util : util + currUtil;
                    itemUtil.put(item, currUtil);
                }
                tempDb.add(transaction);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        ArrayList<Integer> utils = new ArrayList<>(itemUtil.values());
        Collections.sort(utils, Collections.reverseOrder());
        int minUtil = utils.get(k);
        System.out.println("MinUtil:" + minUtil);


        //2nd DB-scan: remove items with TWU < minUtil
        for (int i = 0; i < tempDb.size(); i++) {
            List<Pair> revisedTransaction = new ArrayList<>();
            for (int j = 0; j < tempDb.get(i).size(); j++) {
                Pair pair = tempDb.get(i).get(j);
                if (itemTWU1.get(pair.item) >= minUtil) {
                    revisedTransaction.add(pair);
                } else {
                    int TU = transUtils.get(i) - pair.utility;
                    transUtils.set(i, TU); //update transaction utility since item is removed
                }
            }
            db.add(revisedTransaction); //store revised transaction
        }
        ETP(db, transUtils);

    }

    private void init() {
        Map<Integer, Integer> itemTWU1 = new HashMap<>(); //holds current TWU-value for each item
        List<Integer> transUtils = new ArrayList<>(); //holds TU-value for each transaction
        List<List<Pair>> db = new ArrayList<>();
        Map<Integer, Integer> totalItemUtil = new HashMap<>(); //used to determine minUtil

        String currentLine;
        try (BufferedReader data = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataPath)))) {
            //1st DB-Scan: calculate TWU value for each item
            while ((currentLine = data.readLine()) != null) {
                String[] split = currentLine.split(":");
                String[] items = split[0].split(" ");
                String[] utilities = split[2].split(" ");
                int transactionUtility = Integer.parseInt(split[1]);
                totalUtil += transactionUtility;
                transUtils.add(transactionUtility);
                for (int i = 0; i < items.length; i++) {
                    int item = Integer.parseInt(items[i]);
                    int util = Integer.parseInt(utilities[i]);
                    Integer twu = itemTWU1.get(item);
                    twu = (twu == null) ? transactionUtility : twu + transactionUtility;
                    itemTWU1.put(item, twu);
                    //calculate utility of size 1 itemsets
                    Integer currUtil = totalItemUtil.get(item);
                    currUtil = (currUtil == null) ? util : util + currUtil;
                    totalItemUtil.put(item, currUtil);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        ArrayList<Integer> utils = new ArrayList<>(totalItemUtil.values()); //TODO: Better way??
        Collections.sort(utils, Collections.reverseOrder());
        minUtil = (k < utils.size()) ? utils.get(k) : utils.get(utils.size() - 1);
        System.out.println("MinUtil:" + minUtil);


        try (BufferedReader data = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataPath)))) {
            //2nd DB-Scan: prune
            int tid = 0;
            while ((currentLine = data.readLine()) != null) {
                String[] split = currentLine.split(":");
                String[] items = split[0].split(" ");
                String[] utilities = split[2].split(" ");
                List<Pair> transaction = new ArrayList<>();
                for (int i = 0; i < items.length; i++) {
                    int item = Integer.parseInt(items[i]);
                    int util = Integer.parseInt(utilities[i]);
                    if (itemTWU1.get(item) >= minUtil) {
                        transaction.add(new Pair(item, util));
                    } else {
                        int TU = transUtils.get(tid) - util;
                        transUtils.set(tid, TU);
                    }

                }
                db.add(transaction);
                tid++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        ETP(db, transUtils);
    }


    private void ETP2(List<List<Pair>> db, List<Integer> transUtils) {
        while (true) {
            Map<Integer, Integer> itemTWU1 = new HashMap<>();
            boolean pruned = false;
            for (int i = 0; i < db.size(); i++) {
                int transactionUtility = transUtils.get(i);
                for (int j = 0; j < db.get(i).size(); j++) {
                    int item = db.get(i).get(j).item;
                    Integer twu = itemTWU1.get(item);
                    twu = (twu == null) ? transactionUtility : twu + transactionUtility;
                    itemTWU1.put(item, twu);
                }
            }
            //check if any item has TWU < minUtil
            for (int i = 0; i < db.size(); i++) {
                List<Pair> revisedTransaction = new ArrayList<>();
                for (int j = 0; j < db.get(i).size(); j++) {
                    int item = db.get(i).get(j).item;
                    int twu = itemTWU1.get(item);
                    if (twu >= minUtil) {
                        revisedTransaction.add(db.get(i).get(j)); //add item to revised transaction
                    } else { // item is 1-LTWUI
                        pruned = true;
                        int TU = transUtils.get(i) - db.get(i).get(j).utility;
                        transUtils.set(i, TU); //update transaction utility
                    }
                }
                db.set(i, revisedTransaction);
            }
            if (!pruned) {
                optimizeTransactions(db, itemTWU1);
                break;
            }
        }
    }

    /**
     * Recursively calculates item-TWUs, removes 1-LTWUI and updates TUs, until no items are removed.
     *
     * @param db         The database to prune
     * @param transUtils The current transaction utilities of the database
     */
    private void ETP(List<List<Pair>> db, List<Integer> transUtils) {
        Map<Integer, Integer> itemTWU1 = new HashMap<>();
        boolean pruned = false;
        //calculate TWU of each item
        for (int i = 0; i < db.size(); i++) {
            int transactionUtility = transUtils.get(i);
            for (int j = 0; j < db.get(i).size(); j++) {
                int item = db.get(i).get(j).item;
                Integer twu = itemTWU1.get(item);
                twu = (twu == null) ? transactionUtility : twu + transactionUtility;
                itemTWU1.put(item, twu);
            }
        }
        //check if any item has TWU < minUtil
        for (int i = 0; i < db.size(); i++) {
            List<Pair> revisedTransaction = new ArrayList<>();
            for (int j = 0; j < db.get(i).size(); j++) {
                int item = db.get(i).get(j).item;
                int twu = itemTWU1.get(item);
                if (twu >= minUtil) {
                    revisedTransaction.add(db.get(i).get(j)); //add item to revised transaction
                } else { // item is 1-LTWUI
                    pruned = true;
                    int TU = transUtils.get(i) - db.get(i).get(j).utility;
                    transUtils.set(i, TU); //update transaction utility
                }
            }
            db.set(i, revisedTransaction);
        }
        if (pruned) { //item was removed, repeat pruning
            ETP(db, transUtils);
        } else { //pruning is finished, optimize DB
            optimizeTransactions(db, itemTWU1);

        }
    }

    /**
     * Sets item-names in the range 1 - #1-HTWUI, removes empty transactions,
     * and initializes values required for the fitness- calculation and estimate approach
     * The revised db is stored in 'database'
     *
     * @param db The database to optimize
     */
    private void optimizeTransactions(List<List<Pair>> db, Map<Integer, Integer> itemTWU1) {
        HashMap<Integer, Integer> itemNames = new HashMap<>();
        int c = 0; //new item name
        int transID = 0; //current TID
        for (int i = 0; i < db.size(); i++) {
            if (db.get(i).isEmpty()) {
                continue;
            }
            for (int j = 0; j < db.get(i).size(); j++) {
                int item = db.get(i).get(j).item;
                int utility = db.get(i).get(j).utility;
                if (!itemNames.containsKey(item)) {
                    //item has not been given new name yet
                    c++; //increment name
                    itemNames.put(item, c); //set name for this item
                    itemNamesRev.put(c, item); //save the old name so it can be retrieved later
                    Item itemClass = new Item(c); //this class stores different info for the item
                    items.add(itemClass);
                }
                int twu = itemTWU1.get(item); //get the twu of this item
                item = itemNames.get(item); //get the new name of the item
                db.get(i).get(j).item = item; //change the name of the item
                Item it = items.get(item - 1);
                it.TIDS.set(transID); //update the transaction bit for the item
                itemTWU.put(item, twu); //store twu value
                it.totalUtil += utility; //update total utility of this item
                it.maxUtil = (it.maxUtil == 0) ? utility : Math.max(it.maxUtil, utility); //update max utility
            }
            Collections.sort(db.get(i)); //sort transaction according to item name
            maxTransactionLength = Math.max(maxTransactionLength, db.get(i).size()); //update max trans. length
            database.add(db.get(i)); //store the transaction
            transID++;
        }
    }


    private void writeOut() throws IOException {
        StringBuilder sb = new StringBuilder();
        for (Particle p : sols.getSol()) {
            for (int i = p.X.nextSetBit(0); i != -1; i = p.X.nextSetBit(i + 1)) {
                sb.append(itemNamesRev.get(i));
                sb.append(" ");
            }
            sb.append("#UTIL: ");
            sb.append(p.fitness);
            sb.append(System.lineSeparator());
            disUtil += p.fitness;
        }
        BufferedWriter w = new BufferedWriter(new FileWriter(resultPath));
        w.write(sb.toString());
        w.newLine();
        w.close();
    }

    /*
    private void writeRes() throws IOException {
        StringBuilder sb = new StringBuilder();
        String p;
        if (prune) {
            p = "_PRUNE";
        } else {
            p = "_NOPRUNE";
        }
        for (int i = 0; i < it.size(); i++) {
            sb.append(it.get(i));
            sb.append(",");
            sb.append(pat.get(i));
            sb.append(System.lineSeparator());
        }
        BufferedWriter w = new BufferedWriter(new FileWriter(convPath + "convergence" + p + ".csv"));
        w.write(sb.toString());
        w.newLine();
        w.close();

        sb = new StringBuilder();
        sb.append(minUtil + "," + (minUtil * 1.0 / totalUtil) + "," + chuis.size() + "," + (endTimestamp - startTimestamp) + "," + (int) maxMemory);
        BufferedWriter s = new BufferedWriter(new FileWriter(convPath + "log_" + p + ".csv", true));
        s.write(sb.toString());
        s.newLine();
        s.close();
    }
     */

    /**
     * Print statistics about the latest execution to System.out.
     */
    public void printStats() {
        System.out
                .println("============= STATS =============");
        System.out.println(" Total time ~ " + (endTimestamp - startTimestamp)
                + " ms");
        System.out.println(" Memory ~ " + maxMemory + " MB");
        System.out.println(" Discovered Utility: " + disUtil);
        System.out
                .println("===================================================");
    }

    private void checkMemory() {
        double currentMemory = (Runtime.getRuntime().totalMemory() - Runtime
                .getRuntime().freeMemory()) / 1024d / 1024d;
        maxMemory = Math.max(maxMemory, currentMemory);
    }
}