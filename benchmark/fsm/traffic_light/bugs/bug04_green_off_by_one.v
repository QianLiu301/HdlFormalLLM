// traffic_light — MUTANT (bug04_green_off_by_one.v) — GREEN lasts G_TICKS+1 cycles (boundary off by one)
module traffic_light #(
    parameter G_TICKS = 8,
    parameter Y_TICKS = 3,
    parameter R_TICKS = 6
) (
    input  wire clk,
    input  wire rst_n,
    output wire red,
    output wire yellow,
    output wire green
);
    localparam S_RED = 2'd0, S_GREEN = 2'd1, S_YELLOW = 2'd2;
    reg [1:0] state;
    reg [7:0] cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_RED;
            cnt   <= 8'd0;
        end else begin
            case (state)
                S_RED:    if (cnt == R_TICKS - 1) begin state <= S_GREEN;  cnt <= 8'd0; end
                          else cnt <= cnt + 8'd1;
                S_GREEN:  if (cnt == G_TICKS) begin state <= S_YELLOW; cnt <= 8'd0; end  // BUG: off by one
                          else cnt <= cnt + 8'd1;
                S_YELLOW: if (cnt == Y_TICKS - 1) begin state <= S_RED;    cnt <= 8'd0; end
                          else cnt <= cnt + 8'd1;
                default:  begin state <= S_RED; cnt <= 8'd0; end
            endcase
        end
    end

    assign red    = (state == S_RED);
    assign yellow = (state == S_YELLOW);
    assign green  = (state == S_GREEN);
endmodule
