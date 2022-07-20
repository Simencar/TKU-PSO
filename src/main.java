import java.io.IOException;
import java.util.ArrayList;

public class main {

    public static void main(String[] args) throws IOException {
        TOPK_PSO alg = new TOPK_PSO();
        alg.run();
        alg.printStats();


    }


}
