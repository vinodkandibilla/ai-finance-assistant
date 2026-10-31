import pytest
from langchain_core.messages import AIMessage

from finassist.agents.portfolio_analysis.authorization import UserContext
from finassist.agents.portfolio_analysis.presentation import format_snapshot, is_snapshot_request
from finassist.agents.portfolio_analysis.tools import PortfolioTools
from finassist.data.portfolio_store import PortfolioStore
from test_finance_autonomous_agent import call
from test_portfolio_agent import make_agent


@pytest.mark.parametrize('name,identifier', [('Emma', 'C004'), ('William', 'C003')])
async def test_routine_snapshot_has_consistent_sections(name, identifier):
    with PortfolioStore() as store:
        agent, _ = make_agent(store, [
            call('get_customer_portfolio', {'customer_id': identifier}, 'p'),
            AIMessage(content='Done'),
        ])
        result = await agent.ainvoke({'query': f'Show portfolio of {name}.'})
        assert result['status'] == 'answered'
        for heading in ['Portfolio overview', 'Holdings', 'Asset allocation',
                        'Sector allocation', 'Concentration']:
            assert f'### {heading}' in result['answer']
        assert 'frozen reference prices as of 2026-10-09' in result['answer']
        assert 'Reference price' in result['answer']
        assert identifier in result['answer']


def test_formatter_preserves_missing_costs_as_unavailable():
    with PortfolioStore() as store:
        tools = PortfolioTools(store, user_context=UserContext(
            user_id='PM001', role='PORTFOLIO_MANAGER'))
        snapshot = tools.get_customer_portfolio('C004')
        snapshot['cost_basis'] = None
        snapshot['unrealized_gain_loss'] = None
        assert 'Cost basis: Unavailable' in format_snapshot(snapshot)


@pytest.mark.parametrize('query', ['Compare Emma and William',
                                  'Show portfolio of Emma and analyze risk',
                                  'Show biggest losers for Emma'])
def test_analytical_requests_keep_model_response(query):
    assert not is_snapshot_request(query)
