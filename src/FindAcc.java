import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.TreeSet;


public class FindAcc {

    public static void main(String[] args) {
        String dataset = "mushroom2000";
        String res = "D:\\Documents\\Skole\\Master\\Experiments\\TOPK\\Solutions\\"+dataset+".txt";
        String out = "D:\\Documents\\Skole\\Master\\Work\\out.txt";
        int solutions = 0;
        int found = 0;

        HashSet<TreeSet<Integer>> sol = new HashSet<>();
        HashSet<TreeSet<Integer>> dup = new HashSet<>();


        try (BufferedReader data = new BufferedReader(new InputStreamReader(
                new FileInputStream(res)))) {
            String currentLine;
            while ((currentLine = data.readLine()) != null) {

                String [] split = currentLine.split("#");
                String [] items = split[0].split(" ");
                TreeSet<Integer> itemset = new TreeSet<>();
                for (int i = 0; i < items.length; i++) {
                    itemset.add(Integer.parseInt(items[i]));
                }
                sol.add(itemset);
                if(!itemset.isEmpty()) {
                    solutions++;

                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        try (BufferedReader data = new BufferedReader(new InputStreamReader(
                new FileInputStream(out)))) {
            String currentLine;
            while ((currentLine = data.readLine()) != null) {
                String [] split = currentLine.split("#");
                String [] items = split[0].split(" ");
                TreeSet<Integer> itemset = new TreeSet<>();
                for (int i = 0; i < items.length; i++) {
                    if (!items[i].equals("")) {
                        itemset.add(Integer.parseInt(items[i]));
                    }
                }
                if(sol.contains(itemset)) {
                    found++;
                }
                if (dup.contains(itemset)) {
                    System.out.println("PROBLEM!!!!");
                }
                dup.add(itemset);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        double acc = (((double) found) / solutions) * 100;
        System.out.println("Solutions: " +solutions);
        System.out.println("Found:     " +found);
        System.out.println("Accuracy:  " +acc + " %");

    }
}
