package construct;

import java.util.Arrays;
import java.util.Collection;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Lucene {@link QueryParser} generating alternative underlying queries by sampling from a {@link MultiLayerNetwork}
 * (e.g. charLSTM).
 */
public class ExpandQuery extends QueryParser {

    public ExpandQuery(String field, Analyzer a) {
        super(field, a);
    }

    @Override
    public Query parse(String query) throws ParseException {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        builder.add(new BooleanClause(super.parse(query), BooleanClause.Occur.MUST));

        Collection<String> samples =  fakeNetwork(query);

        for (String sample : samples) {
            builder.add(new BooleanClause(super.parse(sample), BooleanClause.Occur.SHOULD));
        }

        return builder.build();
    }

    private Collection<String> fakeNetwork(String query) {
        switch (query) {
            case "shower":
                //exec(puython.....
                return Arrays.asList("bath", "soap", "dispenser");
            default:
                return Arrays.asList();
        }
    }
}
